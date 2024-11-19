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
from tqdm import tqdm

def compute_metrics(x, recon_x, l1_coefficient=0.001):
    mse_loss = jnp.mean(jnp.square(recon_x - x))
    # mae_loss = jnp.mean(jnp.abs(x - recon_x))

    # return {
    #     'mse_loss': mse_loss,
    #     'mae_loss': mae_loss,
    # }

def train_step(state, batch, z_rng, latent_dim, l1_coefficient=0.001):
    def loss_fn(params):
        fmri_voxels = batch.shape[1]
        recon_x, latent_vec = models.model(latent_dim, fmri_voxels).apply(
            {'params': params}, batch, z_rng
        )

        # loss is composed of reconstruction and sparsity loss,
        # where the sparsity loss uses a l1 coefficient to punish the latent rep activations or sth like that
        mse_loss = jnp.mean(jnp.square(recon_x - batch))
        l1_loss = l1_coefficient * jnp.sum(jnp.abs(latent_vec))
        total_loss = mse_loss + l1_loss

        # can also be impemented as KL-divergence
        # mean_activation = jnp.mean(hidden_activations, axis=0)
        # kl_divergence = jnp.sum(
        #        sparsity_level * jnp.log(sparsity_level / (mean_activation + 1e-10)) +
        #        (1 - sparsity_level) * jnp.log((1 - sparsity_level) / (1 - mean_activation + 1e-10))
        #    )
        return total_loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

def eval_f(params, batch, z_rng, latents):
    def eval_model(ae):
        recon_x, latent_vec = ae(batch, z_rng)

        # get first 100 voxels of the original fmri and reconstructed one
        # NOW: only of the first element in the batch
        # TODO: for all the bathc using etsch visualisation
        recon_as_image = recon_x[:, :100]
        original_as_image = batch[:, :100]
        difference = original_as_image-recon_as_image

        metrics = compute_metrics(batch, recon_x)
        return metrics, difference

    fmri_voxels = batch.shape[1]
    return nn.apply(eval_model, models.model(latents, fmri_voxels))({'params': params})

# def debug_reconstruction(config):
#     """Detailed diagnostics of reconstruction"""
#     # Similar to train_and_evaluate, but with more logging
#     rng = random.key(0)
#     rng, key = random.split(rng)

#     # Load data
#     idxs = nsd_data.split_idxs()
#     subject_idxs = (idxs['subject_train'], idxs['subject_test'])
#     train_loader, test_loader, train_size, test_size, fmri_voxels = nsd_data.create_loaders(
#         subject_idxs, roi=config.roi, hem=config.hem, batch_size=config.batch_size
#     )

#     # Initialize model
#     init_data = jax.random.normal(rng, (config.batch_size, fmri_voxels)).astype(jnp.float32)
#     params = models.model(config.latent_dim, fmri_voxels).init(key, init_data, rng)['params']

#     # Take first batch
#     test_ds = iter(test_loader)
#     test_batch = next(test_ds)

#     # Compute metrics for raw data
#     model_instance = models.model(config.latent_dim, fmri_voxels)
#     recon_x, latent_vec = model_instance.apply({'params': params}, test_batch, rng)

#     metrics = compute_metrics(test_batch, recon_x)

#     # Detailed logging
#     print("Data Diagnostics:")
#     print(f"Input Batch Shape: {test_batch.shape}")
#     print(f"Reconstruction Shape: {recon_x.shape}")
#     print(f"Latent Vector Shape: {latent_vec.shape}")
#     print("\nInitial Metrics:")
#     for k, v in metrics.items():
#         print(f"{k}: {v}")

#     # Visualization
#     plt.figure(figsize=(15,5))
#     plt.subplot(131)
#     plt.title('Original Data')
#     plt.imshow(test_batch, aspect='auto', cmap='viridis')
#     plt.colorbar()

#     plt.subplot(132)
#     plt.title('Reconstructed Data')
#     plt.imshow(recon_x, aspect='auto', cmap='viridis')
#     plt.colorbar()

#     plt.subplot(133)
#     plt.title('Difference')
#     plt.imshow(test_batch - recon_x, aspect='auto', cmap='RdBu_r')
#     plt.colorbar()

#     plt.tight_layout()
#     plt.savefig('reconstruction_debug.png')
#     plt.close()

# def evaluate(state, test_loader, config, fmri_voxels, z_rng):
#     model = models.model(config.latent_dim, fmri_voxels)
#     metrics_sum = {'mse_loss': 0.0, 'mae_loss': 0.0}
#     count = 0

#     for batch in test_loader:
#         recon_x, _ = model.apply({'params': state.params}, batch, z_rng )
#         metrics = compute_metrics(batch, recon_x)

#         for k, v in metrics.items():
#             metrics_sum[k] += v
#         count += 1

#     return {k: v / count for k, v in metrics_sum.items()}

def train_and_evaluate(config):
    """Train and evaulate pipeline."""
    rng = random.key(0)
    rng, key = random.split(rng)

    log('Initializing dataset...', 'TRAIN')
    idxs = nsd_data.split_idxs()
    subject_idxs = (idxs['subject_train'], idxs['subject_test'])
    train_loader, test_loader, train_size, test_size, fmri_voxels = nsd_data.create_loaders(subject_idxs, roi=config.roi, hem=config.hem, batch_size=config.batch_size)

    log('Initializing model...', 'TRAIN')
    init_data = jnp.ones((config.batch_size, fmri_voxels), jnp.float32)

    log('Initializing params...', 'TRAIN')
    params = models.model(config.latent_dim, fmri_voxels).init(key, init_data, rng)['params']

    log('Initializing state...', 'TRAIN')
    state = train_state.TrainState.create(
        apply_fn=models.model(config.latent_dim, fmri_voxels).apply, # calls the fun __call__ in model
        params=params,
        tx=optax.adam(config.learning_rate),
    )
    rng, z_key, eval_rng = random.split(rng, 3)
    z = random.normal(z_key, (64, config.latent_dim)) # test size?

    log('Calculating training steps per epochs...', 'TRAIN')
    steps_per_epoch = train_size // int(config.batch_size)
    if train_size % int(config.batch_size) != 0:
        steps_per_epoch += 1

    # i only need one batch to test agains each epoch
    log('Gathering testing batches...', 'TRAIN')
    test_ds = iter(test_loader)
    test_batch = next(test_ds)

    log('Debugging metrics', 'METRICS')
    # debug_reconstruction(config)

    test_batches = []
    for i, batch in enumerate(test_loader):
        if i >= config.num_epochs:
            break
        test_batches.append(batch)


    log('\nstarting training', 'TRAIN')
    for epoch in range(config.num_epochs):
        log(f'Epoch {epoch + 1}/{config.num_epochs}', 'TRAIN LOOP')

        # Training loop
        epoch_loss = 0.0
        for step, batch in tqdm(enumerate(train_loader), total=steps_per_epoch, desc=f'Training epoch {epoch + 1}'):
            if step >= steps_per_epoch:
                break

            rng, key = random.split(rng)
            state = train_step(state=state, batch=batch, z_rng=key, latent_dim=config.latent_dim)

        # eval - always against the same test batch

        # eval_metrics = evaluate(state, test_loader, config, fmri_voxels, rng)
        # print(f'Epoch {epoch + 1}: '
        #       f'MSE Loss: {eval_metrics["mse_loss"]:.4f}, '
        #       f'MAE Loss: {eval_metrics["mae_loss"]:.4f}')


        # fig, axs = plt.subplots(figsize=(4, 8))
        # axs.imshow(difference, cmap='gray')
        # axs.set_title('Original fMRI + reconstruction')
        # fig.savefig(f'results/difference_{epoch}.png')
        # plt.close()

        # fig2, axs2 = plt.subplots(figsize=(4, 8))
        # axs2.plot(latent_vec)
        # print(latent_vec)
        # axs2.set_title('latent vector View')
        # fig2.savefig(f'results/latent_vec_{epoch}.png')
        # plt.close()

    debug_reconstruction(config)
