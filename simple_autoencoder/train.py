"""Training and evaluation logic."""
import matplotlib.pyplot as plt
import numpy as np
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
    print(loss)
    return {'loss': loss }


def train_step(state, batch, z_rng, latents, l1_coefficient=0.01):
    def loss_fn(params):
        recon_x, latent_vec = models.model(latents, FMRI_DIMENSION).apply(
            {'params': params}, batch, z_rng
        )

        # loss is composed of reconstruction and sparsity loss, where the sparsity loss uses a l1 coefficient to punish the latent rep activations or sth like that
        recon_loss = jnp.mean(jnp.square(recon_x - batch))
        sparsity_loss = l1_coefficient * jnp.sum(latent_vec)
        return recon_loss + sparsity_loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

def eval_f(params, batch, z, z_rng, latents):
    def eval_model(ae):
        recon_x, latent_vec = ae(batch, z_rng)
        print(latent_vec.shape)
        # vec_as_image = np.hstack([np.zeros(86), latent_vec[0], np.zeros(67)])

        recon_as_image = np.reshape(recon_x[0], (173, 42))
        original_as_image = np.reshape(batch[0], (173, 42))

        comparison = np.concatenate([
            original_as_image,
            np.ones((173, 42)),
            # vec_as_image,
            # np.ones((173, 42)),
            recon_as_image,
        ], axis=1)

        # generate_images = ae.generate(z)
        # generate_images = generate_images.reshape(-1, 28, 28, 1)
        metrics = compute_metrics(recon_x, batch)
        return metrics, comparison

    return nn.apply(eval_model, models.model(latents, FMRI_DIMENSION))({'params': params})

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
    test_ds = iter(test_loader)

    print('starting training')
    for epoch in range(config.num_epochs):
        print(f'Epoch {epoch + 1}/{config.num_epochs}')

        # Training loop
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                # reset test ds iterator
                test_ds = iter(test_loader)
                break

            rng, key = random.split(rng)
            state = train_step(state, batch, key, config.latents)

        # eval
        test_batch = next(test_ds)
        metrics, comparison = eval_f(
            state.params, test_batch, z, eval_rng, config.latents
        )

        # TODO: maybe visualise the weights
        # print(state.params)
        # for k,v in state.params.values():
        #     print(k)
        #     print(v.shape)

        plt.figure(figsize=(4, 8))
        plt.imshow(comparison, cmap='gray')
        plt.title('Original fMRI + reconstruction')
        plt.savefig(f'results/fmri_visualization_{epoch}.png')
        plt.close()

        print(
            'eval epoch: {}, loss: {:.4f}'.format(
                epoch + 1, metrics['loss']
            )
        )
