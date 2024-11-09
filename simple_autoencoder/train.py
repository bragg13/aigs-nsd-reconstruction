"""Training and evaluation logic."""

from absl import logging
from flax import linen as nn
import input_pipeline
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

# @jax.vmap
# def binary_cross_entropy_with_logits(logits, labels):
#   logits = nn.log_sigmoid(logits)
#   return -jnp.sum(
#       labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits))
#   )


def compute_metrics(recon_x, x):
    # loss = binary_cross_entropy_with_logits(recon_x, x).mean()
    loss = jnp.mean(jnp.square(recon_x - x))
    return {'loss': loss }


def train_step(state, batch, z_rng, latents):
    def loss_fn(params):
        recon_x = models.model(latents).apply(
            {'params': params}, batch, z_rng
        )

        loss = jnp.mean(jnp.square(recon_x - batch))
        # loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
        return loss

        grads = jax.grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads)


def eval_f(params, images, z, z_rng, latents):
    def eval_model(ae):
        recon_images = ae(images, z_rng)
        comparison = jnp.concatenate([
            images[:8].reshape(-1, 28, 28, 1),
            recon_images[:8].reshape(-1, 28, 28, 1),
        ])

        generate_images = ae.generate(z)
        generate_images = generate_images.reshape(-1, 28, 28, 1)
        metrics = compute_metrics(recon_images, images)
        return metrics, comparison, generate_images

    return nn.apply(eval_model, models.model(latents))({'params': params})


def train_and_evaluate(config: ml_collections.ConfigDict):
    """Train and evaulate pipeline."""
    rng = random.key(0)
    rng, key = random.split(rng)

    print('Initializing dataset.')
    idxs = nsd_data.shuffle_idxs()
    train_loader, test_loader = nsd_data.create_loaders(idxs, config.batch_size)
    train_ds = iter(train_loader)
    test_ds = iter(test_loader)

    # 784 -> initial input length of fmri
    # init_data = jnp.ones((config.batch_size, 784), jnp.float32)
    print('Initializing model.')
    init_data = jnp.ones((config.batch_size, 224, 224, 3), jnp.float32)  # Images
                # jnp.ones((config.batch_size, 19004), jnp.float32),       # fMRI left hemisphere
                # jnp.ones((config.batch_size, 20544), jnp.float32))       # fMRI right hemisphere

    print('initialising params')
    params = models.model(config.latents).init(key, init_data, rng)['params']

    print('Initializing state')
    state = train_state.TrainState.create(
        apply_fn=models.model(config.latents).apply,
        params=params,
        tx=optax.adam(config.learning_rate),
    )
    print('saved')

    rng, z_key, eval_rng = random.split(rng, 3)
    z = random.normal(z_key, (64, config.latents))

        # ds_builder.info.splits['train'].num_examples
    steps_per_epoch = (600 // int(config.batch_size)) # n examples)

    print('starting training')
    for epoch in range(config.num_epochs):
        for s in range(steps_per_epoch):
            # images, lh_fmri, rh_fmri = next(train_ds)
            images = next(train_ds)
            rng, key = random.split(rng)
            state = train_step(state, images, key, config.latents)

            metrics, comparison, sample = eval_f(
                state.params, test_ds, z, eval_rng, config.latents
            )
            vae_utils.save_image(
                comparison, f'results/reconstruction_{epoch}.png', nrow=8
            )
            vae_utils.save_image(sample, f'results/sample_{epoch}.png', nrow=8)

            print(
                'eval epoch: {}, loss: {:.4f}'.format(
                    epoch + 1, metrics['loss']
                )
            )
