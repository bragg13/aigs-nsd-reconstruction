"""Training and evaluation logic."""
import matplotlib.pyplot as plt
from flax import linen as nn
import models
from logger import log
import jax
from jax import random
import jax.numpy as jnp
import optax
from tqdm import tqdm
from nsd_data import get_train_test_datasets, get_batches
from visualisations import plot_losses
from flax.training import train_state

def compute_metrics(recon_x, x):
    loss = jnp.mean(jnp.square(recon_x - x))
    return {'loss': loss }

def train_step(state, batch, key, latent_dim, l1_coefficient=0.01):
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
    def loss_fn(params, key):
        fmri_voxels = batch.shape[1]
        recon_x, latent_vec = models.model(latent_dim, fmri_voxels).apply(
            {'params': params}, batch, key
        )
        mse_loss = jnp.mean(jnp.square(recon_x - batch))
        # sparsity_loss = l1_coefficient * jnp.sum(latent_vec) # jnp.sum(jnp.abs(latent_vec))

        return mse_loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params, key)
    return state.apply_gradients(grads=grads), loss

def evaluate_fun(params, loader, key, latent_dim, num_steps=2):
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
    def eval_model(ae):
        reconstruction, latent_vecs = ae(batch, key)
        metrics = compute_metrics(reconstruction, batch)
        return metrics, (batch, reconstruction), latent_vecs

    evaluations = []
    for _ in range(num_steps):
        batch = next(loader)
        fmri_voxels = batch.shape[1]
        eval = nn.apply(eval_model, models.model(latent_dim, fmri_voxels))({'params': params})
        evaluations.append(eval)
    return evaluations

def train_and_evaluate(config):
    """Train and evaulate pipeline."""
    rng = random.key(0)
    rng, key = random.split(rng)

    log('Initializing dataset...', 'TRAIN')
    train_ds, test_ds = get_train_test_datasets(subject=3, roi_class='floc-bodies', hem='lh')
    print(f"training ds shape: {train_ds.shape}")
    print(f"test ds shape: {test_ds.shape}")
    train_loader = get_batches(train_ds, config.batch_size)
    test_loader = get_batches(test_ds, config.batch_size)
    train_size = train_ds.shape[0]
    test_size = test_ds.shape[0]
    fmri_voxels = train_ds.shape[1]


    log('Initializing model...', 'TRAIN')
    # va bene che siano uni o random gaussian noise?
    init_data = jnp.ones((config.batch_size, fmri_voxels), jnp.float32)  # fmri of 4 shown-images, 2000 voxels each

    log('Initializing params...', 'TRAIN')
    params = models.model(config.latent_dim, fmri_voxels).init(key, init_data, rng)['params']

    log('Initializing state...', 'TRAIN')
    state = train_state.TrainState.create(
        apply_fn=models.model(config.latent_dim, fmri_voxels).apply,
        params=params,
        tx=optax.adamw(config.learning_rate),
    )

    rng, z_key, eval_rng = random.split(rng, 3)

    log(f'Calculating training steps per epochs (train_size: {train_size})...', 'TRAIN')
    steps_per_epoch = train_size // int(config.batch_size)
    if train_size % int(config.batch_size) != 0:
        steps_per_epoch += 1
    log(f"{steps_per_epoch} steps for each ({config.num_epochs}) epoch", 'TRAIN')


    log('\nstarting training', 'TRAIN')
    train_losses = []
    eval_losses = []
    test_losses = []

    for epoch in range(config.num_epochs):
        # log(f'Epoch {epoch + 1}/{config.num_epochs}', 'TRAIN LOOP')
        rng, epoch_key = jax.random.split(rng)
        epoch_train_losses = []
        epoch_eval_losses = []

        # Training loop
        for step, batch in (pbar := tqdm(enumerate(train_loader), total=steps_per_epoch)):
            if step >= steps_per_epoch:
                break

            state, loss = train_step(state, batch, epoch_key, config.latent_dim)
            epoch_train_losses.append(loss)

            # every 25% of steps for this epoch, evaluate the model
            if step % (steps_per_epoch // 4) == 0:
                rng, key = jax.random.split(rng)
                evaluations = evaluate_fun(
                    state.params, train_loader, key, config.latent_dim, num_steps=5
                )

                # calculate average loss among evaluations
                eval_loss = sum([eval[0]['loss'] for eval in evaluations]) / len(evaluations)
                epoch_eval_losses.append(eval_loss)

                # only plotting the first evaluation
                metrics, (evaluated_batches, reconstructions), latent_vecs = evaluations[0]
                fig,axs = plt.subplots(5, 2, figsize=(15,15))
                fig.suptitle('Validation set', fontsize=14)
                axs[0, 0].set_title('original')
                axs[0, 1].set_title('reconstructed')

                # plot the first 5 samples of the first batch evaluated
                for i in range(5):
                    axs[i, 0].imshow(evaluated_batches[i][:3600].reshape(36, 100))
                    axs[i, 1].imshow(reconstructions[i][:3600].reshape(36, 100))

                # should I append the eval loss to the training losses?
                # train_losses.append(metrics['loss'])
                fig.savefig(f'./results/epoch_{epoch}_step_{step}.png')

            pbar.set_description(f"step {step} loss: {str(loss)[:5]}")
        train_losses.append(epoch_train_losses)
        eval_losses.append(epoch_eval_losses)


    # evaluate on the test set
    rng, key = jax.random.split(rng)
    evaluations = evaluate_fun(
        state.params, test_loader, key, config.latent_dim, num_steps=test_size // config.batch_size
    )
    metrics, (evaluated_batches, reconstructions), latent_vecs = evaluations[0]
    test_losses = [eval[0]['loss'] for eval in evaluations]

    # plotting only the first 5 samples of the first batch evaluated
    fig, axs = plt.subplots(5, 2, figsize=(15,15))
    fig.suptitle('Testing set', fontsize=14)
    axs[0, 0].set_title('original')
    axs[0, 1].set_title('reconstructed')
    for i in range(5):
        axs[i, 0].imshow(evaluated_batches[i][:3600].reshape(36, 100))
        axs[i, 1].imshow(reconstructions[i][:3600].reshape(36, 100))

    fig.savefig('./results/testing.png')

    plot_losses(train_losses, eval_losses, test_losses)
