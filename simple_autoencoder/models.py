
"""AE model definitions."""

from flax import linen as nn
from jax import random
import jax.numpy as jnp

class Encoder(nn.Module):
  """AE Encoder."""

  latent_dim: int
  fmri_dim: int
  dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, x, dropout_rng, training: bool = True):
    # reduces the dimensionality of the input by a factor 1.5, 3, 6, 12 (?)
    layers_div = [3, 6]
    for i, div in enumerate(layers_div):
        x = nn.Dense(self.fmri_dim // div, name=f'fc{i}')(x)
        x = nn.BatchNorm(
                    use_running_average=not training,
                    momentum=0.9,
                    epsilon=1e-5
                )(x)
        x = nn.gelu(x)
        # x = nn.Dropout(
        #             rate=self.dropout_rate,
        #         )(x, deterministic=not training, rng=dropout_rng)

    # final layer
    x = nn.Dense(self.latent_dim, name=f'fc{len(layers_div)}')(x)
    x = nn.relu(x)
    return x

class Decoder(nn.Module):
  """AE Decoder."""

  fmri_dim: int
  dropout_rate: float = 0.1
  dataset: str = 'fmri'

  @nn.compact
  def __call__(self, z, dropout_rng, training: bool):
    # increases the dimensionality of the input by a factor 1.5, 3, 6, 12 (?)
    layers_div = [6, 3]
    for i, div in enumerate(layers_div):
        z = nn.Dense(self.fmri_dim // div, name=f'fc{i}')(z)
        z = nn.BatchNorm(
                    use_running_average=not training,
                    momentum=0.9,
                    epsilon=1e-5
                )(z)
        z = nn.gelu(z)
        # z = nn.Dropout(
        #             rate=self.dropout_rate,
        #         )(z, deterministic=not training, rng=dropout_rng)

    # final layer
    z = nn.Dense(self.fmri_dim, name=f'fc{len(layers_div)}')(z)

    if self.dataset != 'fmri':
        z = nn.sigmoid(z)

    return z


class AE(nn.Module):
  """Full AE model."""

  latent_dim: int #= 3000
  fmri_dim: int # = 3633 #7266
  dropout_rate: float = 0.1
  dataset: str = 'fmri'

  def setup(self):
    self.encoder = Encoder(self.latent_dim, self.fmri_dim, self.dropout_rate)
    self.decoder = Decoder(self.fmri_dim, self.dropout_rate, self.dataset)

  def __call__(self, x, dropout_rng, training: bool = True):
    latent_vec = self.encoder(x, dropout_rng=dropout_rng, training=training)
    recon_x = self.decoder(latent_vec, dropout_rng=dropout_rng, training=training)
    return recon_x, latent_vec

  def encode(self, x, dropout_rng):
    latent_vec = self.encoder(x, dropout_rng=dropout_rng, training=False)
    return latent_vec


def model(latent_dim, fmri_dim, dataset='fmri', dropout_rate=0.1):
  return AE(latent_dim=latent_dim, fmri_dim=fmri_dim, dataset=dataset, dropout_rate=dropout_rate)
