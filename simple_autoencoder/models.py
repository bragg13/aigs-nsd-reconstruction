
"""VAE model definitions."""

from flax import linen as nn
from jax import random
import jax.numpy as jnp

class Encoder(nn.Module):
  """AE Encoder."""

  latents: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(2048, name='fc1')(x)
    x = nn.relu(x)
    x = nn.Dense(1024, name='fc2')(x)
    x = nn.relu(x)
    x = nn.Dense(self.latents, name='fc3')(x)
    x = nn.relu(x)
    return x


class Decoder(nn.Module):
  """AE Decoder."""

  fmri_dimension: int

  @nn.compact
  def __call__(self, z):
    z = nn.Dense(1024, name='fc1')(z)
    z = nn.relu(z)
    z = nn.Dense(2048, name='fc2')(z)
    z = nn.relu(z)
    z = nn.Dense(self.fmri_dimension, name='fc3')(z)
    return z


class AE(nn.Module):
  """Full AE model."""

  latents: int = 3000
  fmri_dimension: int = 3633 #7266

  def setup(self):
    self.encoder = Encoder(self.latents)
    self.decoder = Decoder(self.fmri_dimension)

  def __call__(self, x, z_rng):
    latent_vec = self.encoder(x)
    recon_x = self.decoder(latent_vec)
    return recon_x, latent_vec

  def generate(self, z):
    return nn.sigmoid(self.decoder(z))

def model(latents, fmri_dimension):
  return AE(latents=latents, fmri_dimension=fmri_dimension)
