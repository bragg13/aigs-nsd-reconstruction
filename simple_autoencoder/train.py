"""Training and evaluation logic."""
import matplotlib.pyplot as plt
from flax import linen as nn
import models
from logger import log
import jax
from jax import random
import jax.numpy as jnp
import optax
import numpy as np
from tqdm import tqdm
from nsd_data import get_train_test_datasets, get_batches, get_train_test_mnist, get_train_test_cifar100
from visualisations import plot_losses, plot_original_reconstruction
from flax.training import train_state
from typing import Any

class TrainState(train_state.TrainState):
    batch_stats: Any

def create_train_state(key, init_data, latent_dim, fmri_voxels, learning_rate):
    """Creates initial `TrainState`."""
    model_cls = models.model(latent_dim, fmri_voxels)
    variables = model_cls.init(
        {'params': key, 'dropout': key},
        init_data,
        dropout_rng=key,
        training=True
    )
    return TrainState.create(
        apply_fn=model_cls.apply,
        params=variables['params'],
        batch_stats=variables['batch_stats'],
        tx=optax.adamw(learning_rate)
    )

def compute_metrics(recon_x, x, latent_vec):
    mse_loss = jnp.mean(jnp.square(recon_x - x))
    return {'loss': mse_loss }

def train_step(state, batch, key, latent_dim, l1_coefficient=0.01):
    key, dropout_rng = jax.random.split(key)
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
    def loss_fn(params):
        fmri_voxels = batch.shape[1]
        variables = {'params': params, 'batch_stats': state.batch_stats}
        (recon_x, latent_vec), new_model_state = models.model(latent_dim, fmri_voxels).apply(
            variables, batch, dropout_rng, training=True, mutable=['batch_stats'], rngs={'dropout': dropout_rng}
        )
        mse_loss = jnp.mean(jnp.square(recon_x - batch))
        # l2_loss = jnp.sum(jnp.square(latent_vec))
        # sparsity_loss = l1_coefficient * jnp.sum(latent_vec) # jnp.sum(jnp.abs(latent_vec))

        return mse_loss, new_model_state

    (loss, new_model_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_model_state['batch_stats'])
    return state, loss


def evaluate_fun(state, loader, key, latent_dim, num_steps=2):
    key, dropout_rng = jax.random.split(key)
    """Evaluation function that computes metrics on batches.
       Evaluates 2 batches by default (during training),
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
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        reconstruction, latent_vecs = models.model(latent_dim, batch.shape[1]).apply(
            variables, batch, dropout_rng=dropout_rng, training=False, mutable=False)
        metrics = compute_metrics(reconstruction, batch, latent_vecs)
        return metrics, (batch, reconstruction), latent_vecs

    evaluations = []
    for j in range(num_steps):
        batch = next(loader)
        # fmri_voxels = batch.shape[1]
        eval = eval_model(batch)
        evaluations.append(eval)
    return evaluations

def train_and_evaluate(config):
    """Train and evaulate pipeline."""
    rng = random.key(0)
    rng, init_key = random.split(rng)

    log('Initializing dataset...', 'TRAIN')
    if config.ds == 'fmri':
        train_ds, test_ds = get_train_test_datasets(subject=3, roi_class='floc-bodies', hem='lh')
    elif config.ds == 'mnist':
        train_ds, validation_ds = get_train_test_mnist()
    else:
        train_ds, validation_ds = get_train_test_cifar100()

    print(f"training ds shape: {train_ds.shape}")
    print(f"test ds shape: {validation_ds.shape}")
    key1, key2 = random.split(rng)
    train_loader = get_batches(train_ds, key1, config.batch_size)
    validation_loader = get_batches(validation_ds, key2, config.batch_size)

    train_size = train_ds.shape[0]
    fmri_voxels = train_ds.shape[1]

    # print(train_ds.min(), train_ds.max())

    log('Initializing model...', 'TRAIN')
    # va bene che siano uni o random gaussian noise?
    init_data = jnp.ones((config.batch_size, fmri_voxels), jnp.float32)

    log('Initializing state...', 'TRAIN')
    state = create_train_state(init_key, init_data, config.latent_dim, fmri_voxels, config.learning_rate)

    log(f'Calculating training steps per epochs (train_size: {train_size})...', 'TRAIN')
    steps_per_epoch = train_size // int(config.batch_size)
    if train_size % int(config.batch_size) != 0:
        steps_per_epoch += 1
    log(f"{steps_per_epoch} steps for each ({config.num_epochs}) epoch", 'TRAIN')


    log('\nstarting training', 'TRAIN')
    train_losses = []
    eval_losses = []

    print("Train data stats:", train_ds.min(), train_ds.max(), train_ds.mean())
    print("Validation data stats:", validation_ds.min(), validation_ds.max(), validation_ds.mean())

    for epoch in range(config.num_epochs):
        # log(f'Epoch {epoch + 1}/{config.num_epochs}', 'TRAIN LOOP')
        rng, epoch_key = jax.random.split(rng)

        # Training loop
        for step, batch in (pbar := tqdm(enumerate(train_loader), total=steps_per_epoch)):
            if step >= steps_per_epoch:
                break

            state, loss = train_step(state, batch, epoch_key, config.latent_dim)

            # save the loss every 5% of steps
            if step % (steps_per_epoch // 50) == 0:
                # add training loss
                train_losses.append(loss)

                rng, key = jax.random.split(rng)
                evaluations = evaluate_fun(
                    state, validation_loader, epoch_key, config.latent_dim, num_steps=1
                )

                # add validation losses
                eval_tmp_losses = [eval[0]['loss'] for eval in evaluations]
                eval_losses.extend(eval_tmp_losses)

                # only plotting the first evaluation
                metrics, (evaluated_batches, reconstructions), latent_vecs = evaluations[0]
                del evaluations

                # once every epoch, plot the original and reconstructed images
                if step % (steps_per_epoch // 1) == 0:
                    plot_original_reconstruction(evaluated_batches, reconstructions, config.ds, epoch, step)

            pbar.set_description(f"epoch {epoch} loss: {str(loss)[:5]}")

    plot_losses(train_losses, eval_losses, steps_per_epoch)

    # plot the latent space
    # final_variables = {'params': state.params, 'batch_stats': state.batch_stats}
    # dropout_rng, key = random.split(rng)
    # latent_vectors = []
    # categories = []
    # for step, batch in enumerate(validation_loader):
    #     _, latent_vec= models.model(config.latent_dim, batch.shape[1]).apply(
    #                 final_variables, batch, dropout_rng=dropout_rng, training=False, mutable=False)
    #     latent_vectors.append(latent_vec)
    #     categories.append(step)
    #     del latent_vec
    # plot_latent_space(latent_vectors, categories)
