# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
load_from_disk = False

@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
  logits = nn.log_sigmoid(logits)
  return -jnp.sum(
      labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits))
  )


def compute_metrics(recon_x, x):
  loss = binary_cross_entropy_with_logits(recon_x, x).mean()
  return {'loss': loss }


def train_step(state, batch, z_rng, latents):
  def loss_fn(params):
    recon_x = models.model(latents).apply(
        {'params': params}, batch, z_rng
    )

    loss = jnp.mean(jnp.square(recon_x - batch))
    loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
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

  # our dataset should be sourced instead of the below
  # ds_builder = tfds.builder('binarized_mnist')
  # ds_builder.download_and_prepare()

  print('Initializing dataset.')
  # train_ds = input_pipeline.build_train_set(config.batch_size, ds_builder)
  # test_ds = input_pipeline.build_test_set(ds_builder)

  # our dataset:
  idxs = nsd_data.shuffle_idxs()
  train_loader, test_loader = nsd_data.create_loaders(idxs, config.batch_size)
  train_ds = iter(train_loader)
  test_ds = iter(test_loader)

  CKPT_DIR = '/Users/andrea/Desktop/aigs-nsd-reconstruction/checkpoints/'
  if not load_from_disk:
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
    checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=state, step=0)
    print('saved')
  else :
    restored_state = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=state)
    state = restored_state
  print('okqy')

  rng, z_key, eval_rng = random.split(rng, 3)
  z = random.normal(z_key, (64, config.latents))

      # ds_builder.info.splits['train'].num_examples
  steps_per_epoch = (600 // int(config.batch_size)) # n examples)

  print('starting training')
  for epoch in range(config.num_epochs):
    print(epoch)
    for s in range(steps_per_epoch):
      print(s)
      images, lh_fmri, rh_fmri = next(train_ds)
      # our data
      # batch = next(iter(train_loader))
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
