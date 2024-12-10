"""Training and evaluation logic."""
import aim
import models
from logger import log
import jax
from jax import random
import jax.numpy as jnp
import optax
from tqdm import tqdm
from nsd_data import (
    get_train_test_datasets,
    get_batches,
    get_train_test_mnist,
    get_train_test_cifar10,
)
from visualisations import (
    plot_latent_heatmap,
    visualize_latent_activations,
    LatentVisualizer,
    plot_losses,
    plot_original_reconstruction_fmri,
)
from flax.training import train_state
from typing import Any
import jaxpruner
import ml_collections
import orbax.checkpoint as ocp
from ae_main import PROJECT_DIR


class TrainState(train_state.TrainState):
    batch_stats: Any


def create_train_state(key, init_data, config, fmri_voxels):
    """Creates initial `TrainState`."""
    model = models.model(config.latent_dim, fmri_voxels, dataset=config.ds)
    variables = model.init(
        {"params": key, "dropout": key}, init_data, dropout_rng=key, training=True
    )

    # introducing sparsity
    sparsity_config = ml_collections.ConfigDict()
    sparsity_config.algorithm = "static_sparse"
    sparsity_config.sparsity = config.sparsity
    sparsity_config.dist_type = "erk"

    sparsity_updater = jaxpruner.create_updater_from_config(sparsity_config)
    tx = optax.adamw(config.learning_rate)
    tx = sparsity_updater.wrap_optax(tx)

    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        batch_stats=variables["batch_stats"],
        tx=tx,
    ), sparsity_updater


def compute_metrics(recon_x, x, latent_vec, config):
    mse_loss = jnp.mean(jnp.square(recon_x - x))
    sparsity_loss = config.l1 * jnp.mean(jnp.abs(latent_vec))
    return {"mse_loss": mse_loss, "sparsity_loss": sparsity_loss}


def train_step(state, batch, key, config):
    """Performs a single training step updating model parameters based on batch data.

    Args:
        state: TrainState object containing model state and optimizer
        batch: Current batch of training data
        key: Random key for generating latent vectors
        latent_dim: Dimension of the latent space
        l1_coefficient: Optional coefficient for L1 regularization (default 0.01)

    Returns:
        Tuple containing updated state and training loss for the batch
    """
    key, dropout_rng = jax.random.split(key)

    def loss_fn(params):
        fmri_voxels = batch.shape[1]
        variables = {"params": params, "batch_stats": state.batch_stats}
        (recon_x, latent_vec), new_model_state = models.model(
            config.latent_dim, fmri_voxels, dataset=config.ds
        ).apply(
            variables,
            batch,
            dropout_rng,
            training=True,
            mutable=["batch_stats"],
            rngs={"dropout": dropout_rng},
        )

        # MSE loss and L1 regularization for sparsity in the latent vector
        mse_loss = jnp.mean(jnp.square(recon_x - batch))
        sparsity_loss = config.l1 * jnp.mean(jnp.abs(latent_vec))
        total_loss = mse_loss + sparsity_loss

        return total_loss, (
            new_model_state,
            {"mse_loss": mse_loss, "sparsity_loss": sparsity_loss},
        )

    (loss, (new_model_state, losses)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_model_state["batch_stats"])
    return state, losses


def evaluate_fun(state, evaluation_batch, key, config):
    key, dropout_rng = jax.random.split(key)
    """Evaluation function that computes metrics on batches.
       Evaluates 1 batch by default (during training),
       while evaluating the entire dataset during testing.

    Args:
        params: Model parameters to evaluate
        train_loader: Generator yielding training batches
        key: Random key for generating latent vectors
        latent_dim: Dimension of the latent space
        num_steps: Number of evaluation steps to perform

    Returns:
        List of tuples containing (metrics, data, latent vectors) for each evaluation step
        Average of the losses
    """

    def eval_model(batch):
        variables = {"params": state.params, "batch_stats": state.batch_stats}
        (reconstruction, latent_vecs), _ = models.model(
            config.latent_dim, batch.shape[1], dataset=config.ds
        ).apply(
            variables,
            batch,
            dropout_rng=dropout_rng,
            training=True,
            mutable=["batch_stats"],
        )
        metrics = compute_metrics(reconstruction, batch, latent_vecs, config)
        return metrics, (batch, reconstruction), latent_vecs

    eval = eval_model(evaluation_batch)
    return eval


def train_and_evaluate(config):
    """Train and evaulate pipeline."""
    rng = random.key(0)
    rng, init_key = random.split(rng)

    # initialise AIM run
    run = aim.Run()

    log("Initializing dataset...", "TRAIN")
    rate_reconstruction_printing = 10
    if config.ds == "fmri":
        train_ds, validation_ds = get_train_test_datasets(
            subject=config.subject, roi_class=config.roi_class, hem=config.hem
        )

    elif config.ds == "mnist":
        train_ds, validation_ds = get_train_test_mnist()
    else:
        train_ds, validation_ds = get_train_test_cifar10()

    print(f"training ds shape: {train_ds.shape}")
    print(f"test ds shape: {validation_ds.shape}")
    key1, key2 = random.split(rng)
    train_loader = get_batches(train_ds, key1, config.batch_size)
    validation_loader = get_batches(validation_ds, key2, config.batch_size)

    train_size = train_ds.shape[0]
    fmri_voxels = train_ds.shape[1]

    log("Initializing model...", "TRAIN")
    init_data = jnp.ones((config.batch_size, fmri_voxels), jnp.float32)

    log("Initializing state...", "TRAIN")
    state, sparsity_updater = create_train_state(
        init_key, init_data, config, fmri_voxels
    )

    log(f"Calculating training steps per epochs (train_size: {train_size})...", "TRAIN")
    steps_per_epoch = train_size // int(config.batch_size)
    if train_size % int(config.batch_size) != 0:
        steps_per_epoch += 1
    log(f"{steps_per_epoch} steps for each ({config.num_epochs}) epoch", "TRAIN")

    log("\nstarting training", "TRAIN")
    train_mse_losses = []
    train_spa_losses = []
    eval_losses = []

    print(
        "Train data stats:", train_loader.min(), train_loader.max(), train_loader.mean(), len(train_loader)
    )
    print(
        "Validation data stats:",
        validation_loader.min(),
        validation_loader.max(),
        validation_loader.mean(),
        len(validation_loader)
    )

    # utils for visaulisation and checkpointing
    visualizer = LatentVisualizer(config.results_folder)
    checkpointer = ocp.StandardCheckpointer()
    train_step_jit = jax.jit(train_step, static_argnums=3)
    evaluate_fun_jit = jax.jit(evaluate_fun, static_argnums=3)

    # init aim hyperparameters
    run['hparams'] = {
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'latent_dim': config.latent_dim,
        'sparsity': config.sparsity,
        'l1': config.l1,
        'num_epochs': config.num_epochs,
        'subject': config.subject,
        'hemisphere': config.hem,
        'roi_class': config.roi_class,
    }

    for epoch in range(config.num_epochs):
        rng, epoch_key = jax.random.split(rng)
        validation_step = 0
        val_loss = 100  # only used for initial tqdm printing

        # im reshuffling also the first time, which is useless but I think the code is cleaner
        key1, key2 = random.split(rng)
        train_loader = get_batches(train_ds, key1, config.batch_size)
        validation_loader = get_batches(validation_ds, key2, config.batch_size)

        # pre_op = jax.jit(sparsity_updater.pre_forward_update)
        post_op = jax.jit(sparsity_updater.post_gradient_update)

        tmp_loss = []

        # Training loop
        for step in (
            pbar := tqdm(
                range(0, len(train_loader), config.batch_size), total=steps_per_epoch
            )
        ):
            batch = train_loader[step : step + config.batch_size]
            state, losses = train_step_jit(state, batch, epoch_key, config)
            mse_loss, spa_loss = losses["mse_loss"], losses["sparsity_loss"]

            # implemening sparsity
            post_params = post_op(state.params, state.opt_state)
            state = state.replace(params=post_params)

            tmp_loss.append(mse_loss)
            # train_mse_losses.append(mse_loss)
            # train_spa_losses.append(spa_loss)

            if step % (steps_per_epoch // 5) == 0:
                validation_batch = validation_loader[
                    validation_step : validation_step + config.batch_size
                ]
                validation_step = +config.batch_size

                metrics, (evaluated_batches, reconstructions), latent_vecs = (
                    evaluate_fun_jit(state, validation_batch, epoch_key, config)
                )

                visualizer.update(latent_vecs)

                # average the training loss and append it to the list
                train_mse_losses.append(jnp.mean(jnp.array(tmp_loss)))
                run.track(jnp.mean(jnp.array(tmp_loss)), name='train_loss')
                run.track(metrics['mse_loss'], name='validation_loss')
                tmp_loss = []

                eval_losses.append(metrics["mse_loss"])
                val_loss = metrics["mse_loss"]
                # print(jaxpruner.summarize_sparsity(
                #             state.params, only_total_sparsity=True))

            pbar.set_description(
                f"epoch {epoch} mse loss: {str(mse_loss)[:5]} spa loss: {str(spa_loss)[:5]} val_loss: {str(val_loss)[:5]}"
            )

        # once every 10 epochs, plot the original and reconstructed images of the last batch
        if epoch % rate_reconstruction_printing == 0:
            validation_batch = validation_loader[
                validation_step : validation_step + config.batch_size
            ]
            validation_step = +config.batch_size
            metrics, (evaluated_batches, reconstructions), latent_vecs = evaluate_fun(
                state, validation_batch, epoch_key, config
            )

            # plot_original_reconstruction(evaluated_batches, reconstructions, config, epoch)
            # visualize_latent_activations(latent_vecs, evaluated_batches, config.results_folder,epoch)
            # plot_latent_heatmap(latent_vecs, evaluated_batches, config.results_folder,epoch)

    # save model to disk TODO: use os for this!!
    ckpt_folder = ocp.test_utils.erase_and_create_empty(
        f"{PROJECT_DIR}/{config.results_folder}/checkpoints"
    )
    checkpointer.save(ckpt_folder / "final", state)
    plot_losses(
        train_mse_losses,
        train_spa_losses,
        config.results_folder,
        eval_losses,
        steps_per_epoch,
    )
    # plot_original_reconstruction_fmri(config.subject, evaluated_batches, reconstructions, config.hem)
    visualize_latent_activations(
        latent_vecs, evaluated_batches, config, epoch
    )
    plot_latent_heatmap(latent_vecs, evaluated_batches, config, epoch)
    visualizer.plot_training_history()
