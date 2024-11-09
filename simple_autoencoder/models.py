
"""VAE model definitions."""

from flax import linen as nn
from jax import random
import jax.numpy as jnp
fmri_dimension = 2000

class Encoder(nn.Module):
  """AE Encoder."""

  latents: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(500, name='fc1')(x)
    x = nn.relu(x)
    x = nn.Dense(self.latents, name='fc2')(x)
    return x


class Decoder(nn.Module):
  """AE Decoder."""

  @nn.compact
  def __call__(self, z):
    z = nn.Dense(500, name='fc1')(z)
    z = nn.relu(z)
    z = nn.Dense(fmri_dimension, name='fc2')(z)
    return z


class AE(nn.Module):
  """Full AE model."""

  latents: int = 20

  def setup(self):
    self.encoder = Encoder(self.latents)
    self.decoder = Decoder()

  def __call__(self, x, z_rng):
    latent_vec = self.encoder(x)
    recon_x = self.decoder(latent_vec)
    return recon_x

  def generate(self, z):
    return nn.sigmoid(self.decoder(z))

def model(latents):
  return AE(latents=latents)
