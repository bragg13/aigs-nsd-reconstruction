"""Training and evaluation logic."""

from absl import logging
from flax import linen as nn
import nsd_data
import models
import utils as vae_utils
from flax.training import train_state, checkpoints
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import optax
import tensorflow_datasets as tfds
FMRI_DIMENSION = 7266


def compute_metrics(recon_x, x):
    # loss = binary_cross_entropy_with_logits(recon_x, x).mean()
    loss = jnp.mean(jnp.square(recon_x - x))
    return {'loss': loss }


def train_step(state, batch, z_rng, latents):
    def loss_fn(params):
        recon_x = models.model(latents, FMRI_DIMENSION).apply(
            {'params': params}, batch, z_rng
        )

        loss = jnp.mean(jnp.square(recon_x - batch))
        return loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


def train_and_evaluate(config: ml_collections.ConfigDict):
    """Train and evaulate pipeline."""
    rng = random.key(0)
    rng, key = random.split(rng)

    print('Initializing dataset.')
    idxs = nsd_data.split_idxs()
    subject_idxs = (idxs['subject_train'], idxs['subject_train'])
    train_loader, test_loader = nsd_data.create_loaders(subject_idxs, roi=None, batch_size=config.batch_size)

    print('Initializing model.')
    init_data = jnp.ones((config.batch_size, 4, FMRI_DIMENSION), jnp.float32)  # fmri of 4 shown-images, 2000 voxels each

    print('initialising params')
    params = models.model(config.latents, FMRI_DIMENSION).init(key, init_data, rng)['params']

    print('Initializing state')
    state = train_state.TrainState.create(
        apply_fn=models.model(config.latents, FMRI_DIMENSION).apply,
        params=params,
        tx=optax.adam(config.learning_rate),
    )

    rng, z_key, eval_rng = random.split(rng, 3)
    z = random.normal(z_key, (64, config.latents))

    train_size = 600
    steps_per_epoch = train_size // int(config.batch_size)
    if train_size % int(config.batch_size) != 0:
        steps_per_epoch += 1

    print('starting training')
    for epoch in range(config.num_epochs):
        print(f'Epoch {epoch + 1}/{config.num_epochs}')

        # Training loop
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break

            rng, key = random.split(rng)
            state = train_step(state, batch, key, config.latents)
