"""Training and evaluation logic."""
import matplotlib.pyplot as plt
import numpy as np
from flax import linen as nn
import nsd_data
import models
from logger import log
import utils as vae_utils
from flax.training import train_state, checkpoints
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import optax
import tensorflow_datasets as tfds
# FMRI_DIMENSION = 3633


def compute_metrics(recon_x, x):
    loss = jnp.mean(jnp.square(recon_x - x))
    return {'loss': loss }


def train_step(state, batch, z_rng, latents, l1_coefficient=0.01):

    def loss_fn(params):
        recon_x, latent_vec = models.model(latents, FMRI_DIMENSION).apply(
            {'params': params}, batch, z_rng
        )

        # loss is composed of reconstruction and sparsity loss,
        # where the sparsity loss uses a l1 coefficient to punish the latent rep activations or sth like that
        mse_loss = jnp.mean(jnp.square(recon_x - batch))
        sparsity_loss = l1_coefficient * jnp.sum(latent_vec)

        # can also be impemented as KL-divergence
        # mean_activation = jnp.mean(hidden_activations, axis=0)
        # kl_divergence = jnp.sum(
        #        sparsity_level * jnp.log(sparsity_level / (mean_activation + 1e-10)) +
        #        (1 - sparsity_level) * jnp.log((1 - sparsity_level) / (1 - mean_activation + 1e-10))
        #    )
        return mse_loss #+ sparsity_loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

def eval_f(params, batch, z_rng, latents):
    def eval_model(ae):
        recon_x, latent_vec = ae(batch, z_rng)

        recon_as_image = np.reshape(recon_x[0], (173, 21))
        original_as_image = np.reshape(batch[0], (173, 21))

        comparison = np.concatenate([
            original_as_image,
            recon_as_image,
            original_as_image-recon_as_image
        ], axis=1)

        metrics = compute_metrics(recon_x, batch)
        return metrics, comparison, latent_vec[0]

    FMRI_DIMENSION = 100
    print(batch.size)
    return nn.apply(eval_model, models.model(latents, FMRI_DIMENSION))({'params': params})

def train_and_evaluate(config):
    """Train and evaulate pipeline."""
    rng = random.key(0)
    rng, key = random.split(rng)

    log('Initializing dataset...', 'TRAIN')
    idxs = nsd_data.split_idxs()
    subject_idxs = (idxs['subject_train'], idxs['subject_test'])
    train_loader, test_loader, train_size, test_size, x_dim = nsd_data.create_loaders(subject_idxs, roi=config.roi, hem=config.hem, batch_size=config.batch_size)

    log('Initializing model...', 'TRAIN')
    init_data = jnp.ones((config.batch_size, x_dim), jnp.float32)

    log('Initializing params...', 'TRAIN')
    params = models.model(config.latent_dim, x_dim).init(key, init_data, rng)['params']

    log('Initializing state...', 'TRAIN')
    state = train_state.TrainState.create(
        apply_fn=models.model(config.latent_dim, x_dim).apply, # calls the fun __call__ in model
        params=params,
        tx=optax.adam(config.learning_rate),
    )

    rng, z_key, eval_rng = random.split(rng, 3)
    z = random.normal(z_key, (64, config.latent_dim)) # test size?

    steps_per_epoch = train_size // int(config.batch_size)
    if train_size % int(config.batch_size) != 0:
        steps_per_epoch += 1
    test_ds = iter(test_loader)

    test_batches = []

    log('starting training', 'TRAIN')
    for epoch in range(config.num_epochs):
        log(f'Epoch {epoch + 1}/{config.num_epochs}', 'TRAIN-LOOP')

        # Training loop
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break

            rng, key = random.split(rng)
            state = train_step(state, batch, key, config.latents)

        # eval
        metrics, comparison, latent_vec = eval_f(
            state.params, test_batches[epoch], z, eval_rng, config.latents
        )

        # TODO: maybe visualise the weights
        # print(state.params)
        # for k,v in state.params.values():
        #     print(k)
        #     print(v.shape)

        fig1, axs1 = plt.subplots(figsize=(4, 8))
        axs1.imshow(comparison, cmap='gray')
        axs1.set_title('Original fMRI + reconstruction')
        fig1.savefig(f'results/fmri_visualization_{epoch}.png')
        plt.close()

        fig2, axs2 = plt.subplots(figsize=(4, 8))
        axs2.plot(latent_vec)
        print(latent_vec)
        axs2.set_title('latent vector View')
        fig2.savefig(f'results/latent_vec_{epoch}.png')
        plt.close()

        print(
            'eval epoch: {}, loss: {:.4f}'.format(
                epoch + 1, metrics['loss']
            )
        )
