"""Training and evaluation logic."""
import matplotlib.pyplot as plt
import numpy as np
from absl import logging
from flax import linen as nn
import nsd_data
import models
from logger import log
from visualisations import plot_first_100vx_over_epochs, plot_losses, plot_results_before_after_training, plot_results_epoch
import utils as vae_utils
from flax.training import train_state, checkpoints
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import optax
import tensorflow_datasets as tfds
from tqdm import tqdm
from nsd_data import get_train_test_datasets, get_batches

def compute_metrics(recon_x, x):
    loss = jnp.mean(jnp.square(recon_x - x))
    return {'loss': loss }

def train_step(state, batch, key, latent_dim, l1_coefficient=0.01):
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

def evaluate_fun(params, train_loader, key, latent_dim, num_steps):
    def eval_model(ae):
        recon_x, latent_vecs = ae(batch, key)
        metrics = compute_metrics(recon_x, batch)
        return metrics, (batch, recon_x), latent_vecs

    evaluations = []
    for _ in range(num_steps):
        batch = next(train_loader)
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
    eval_loader = get_batches(train_ds, config.batch_size)
    train_size = train_ds.shape[0]
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

    # log('Collecting test batches...', 'TRAIN')
    # test_ds = iter(test_loader)
    # test_batches = []
    # for step, batch in enumerate(test_loader):
    #     test_batches.append(batch)
    # log(f'collected {len(test_batches)} test batches', 'TRAIN')

    log('\nstarting training', 'TRAIN')
    losses = []

    for epoch in range(config.num_epochs):
        log(f'Epoch {epoch + 1}/{config.num_epochs}', 'TRAIN LOOP')

        # Training loop
        for step, batch in (pbar := tqdm(enumerate(train_loader), total=steps_per_epoch)):
            if step >= steps_per_epoch:
                break

            state, loss = train_step(state, batch, key, config.latent_dim)
            evaluations = evaluate_fun(
                state.params, train_loader, key, config.latent_dim, num_steps=2
            )
            print(f'evaluations has length {len(evaluations)}')
            metrics, (evaluated_batches, reconstructions), latent_vecs = evaluations[0]

            print('plotting voxels at index 0 from training set')
            fig,axs = plt.subplots(5, 2, figsize=(15,15))
            axs[0, 0].set_title('original')
            axs[0, 1].set_title('reconstructed')
            for i in range(5):
                axs[i, 0].imshow(evaluated_batches[i][:3600].reshape(36, 100))
                axs[i, 1].imshow(reconstructions[i][:3600].reshape(36, 100))
            fig.savefig(f'./results/epoch_{epoch}_step_{step}.png')

            pbar.set_description(f"step {step} loss: {loss}")
            losses.append(loss)

            # if (step % (steps_per_epoch // 100)) == 0:
            #     rng, key = jax.random.split(rng)
            #     # metrics = evaluate(params, key, train_loader, val_loader)
            #     evaluations = evaluate_fun(
            #         state.params, eval_loader, key, config.latent_dim
            #     )
            #     metrics, (evaluated_batches, reconstructions), latent_vecs = evaluations[0]
            #     plot_results_epoch(evaluated_batches, reconstructions, latent_vecs, epoch, step)



        # for evolution visualisation
        # print(f"data to rec has shape {data_to_reconstruct.shape} ")
        # metrics, (batch, reconstructions), latent_vecs = eval_f(
        #     state.params, data_to_reconstruct, z, eval_rng, config.latent_dim
        # )
        # print(f" rec has shape {reconstructions.shape} ")
        # all_reconstructions.append(reconstructions)

    # plot_first_100vx_over_epochs(np.array(all_reconstructions))
    # plot_results_before_after_training(batch reconstructions, latent_vec)
    plot_losses(losses)
