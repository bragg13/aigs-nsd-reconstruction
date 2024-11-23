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


def compute_metrics(recon_x, x):
    loss = jnp.mean(jnp.square(recon_x - x))
    return {'loss': loss }


def train_step(state, batch, z_rng, latent_dim, l1_coefficient=0.01):
    def loss_fn(params):
        fmri_voxels = batch.shape[1]
        # print(batch.shape)
        recon_x, latent_vec = models.model(latent_dim, fmri_voxels).apply(
            {'params': params}, batch, z_rng
        )
        mse_loss = jnp.mean(jnp.square(recon_x - batch))
        sparsity_loss = l1_coefficient * jnp.sum(latent_vec) # jnp.sum(jnp.abs(latent_vec))

        return mse_loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

def eval_f(params, batch, z, z_rng, latent_dim):
    def eval_model(ae):
        recon_x, latent_vecs = ae(batch, z_rng)
        # log(f'dim reconx: {recon_x.shape}', 'EVAL_F')
        metrics = compute_metrics(recon_x, batch)
        return metrics, (batch, recon_x), latent_vecs

    fmri_voxels = batch.shape[1]
    return nn.apply(eval_model, models.model(latent_dim, fmri_voxels))({'params': params})

def train_and_evaluate(config):
    """Train and evaulate pipeline."""
    rng = random.key(0)
    rng, key = random.split(rng)

    log('Initializing dataset...', 'TRAIN')
    subject_dataset_train, subject_dataset_test = nsd_data.get_train_test_datasets(subject=3)
    # train_loader, test_loader = nsd_data.create_loaders(subject_idxs, roi=None, batch_size=config.batch_size)

    log('Initializing model...', 'TRAIN')
    # va bene che siano uni en on random gaussian noise?
    init_data = jnp.ones((config.batch_size, fmri_voxels), jnp.float32)  # fmri of 4 shown-images, 2000 voxels each

    log('Initializing params...', 'TRAIN')
    params = models.model(config.latent_dim, fmri_voxels).init(key, init_data, rng)['params']

    log('Initializing state...', 'TRAIN')
    state = train_state.TrainState.create(
        apply_fn=models.model(config.latent_dim, fmri_voxels).apply,
        params=params,
        tx=optax.adam(config.learning_rate),
    )

    rng, z_key, eval_rng = random.split(rng, 3)
    z = random.normal(z_key, (64, config.latent_dim))

    log(f'Calculating training steps per epochs (train_size: {train_size})...', 'TRAIN')
    steps_per_epoch = train_size // int(config.batch_size)
    if train_size % int(config.batch_size) != 0:
        steps_per_epoch += 1
    log(f"{steps_per_epoch} steps for each ({config.num_epochs}) epoch", 'TRAIN')

    log('Collecting test batches...', 'TRAIN')
    test_ds = iter(test_loader)
    test_batches = []
    for step, batch in enumerate(test_loader):
        test_batches.append(batch)
    log(f'collected {len(test_batches)} test batches', 'TRAIN')

    log('\nstarting training', 'TRAIN')
    losses = []

    data_to_reconstruct = test_batches[0]
    all_reconstructions = []
    all_reconstructions.append(data_to_reconstruct)

    for epoch in range(config.num_epochs):
        log(f'Epoch {epoch + 1}/{config.num_epochs}', 'TRAIN LOOP')

        # Training loop
        for step, batch in tqdm(enumerate(train_loader), total=steps_per_epoch):
            if step >= steps_per_epoch:
                break

            rng, key = random.split(rng)
            state = train_step(state, batch, key, config.latent_dim)
            metrics, (_batch, reconstructions), latent_vecs = eval_f(
                state.params, batch, z, eval_rng, config.latent_dim
            )
            losses.append(metrics['loss'])
            plot_results_epoch(_batch, reconstructions, latent_vecs, epoch)

        # eval
        # metrics, (_batch, reconstructions), latent_vecs = eval_f(
        #     state.params, batch, z, eval_rng, config.latent_dim
        # )
        # metrics, (batch, reconstructions), latent_vecs = eval_f(
        #     state.params, test_batches[epoch], z, eval_rng, config.latent_dim
        # )
        # print(
        #     'eval epoch: {}, loss: {:.4f}'.format(
        #         epoch + 1, metrics['loss']
        #     )
        # )
        # losses.append(metrics['loss'])

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
