
"""VAE model definitions."""

from flax import linen as nn
from jax import random
import jax.numpy as jnp

class Encoder(nn.Module):
  """AE Encoder."""

  latent_dim: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(2048, name='fc1')(x)
    x = nn.relu(x)

    x = nn.Dense(1024, name='fc2')(x)
    x = nn.relu(x)

    x = nn.Dense(self.latent_dim, name='fc3')(x)
    return x

class Decoder(nn.Module):
  """AE Decoder."""

  fmri_dim: int

  @nn.compact
  def __call__(self, z):
    z = nn.Dense(1024, name='fc1')(z)
    z = nn.relu(z)

    z = nn.Dense(2048, name='fc2')(z)
    z = nn.relu(z)

    z = nn.Dense(self.fmri_dim, name='fc3')(z)
    return z


class AE(nn.Module):
  """Full AE model."""

  latent_dim: int #= 3000
  fmri_dim: int # = 3633 #7266

  def setup(self):
    self.encoder = Encoder(self.latent_dim)
    self.decoder = Decoder(self.fmri_dim)

  def __call__(self, x, z_rng):
    latent_vec = self.encoder(x)
    recon_x = self.decoder(latent_vec)
    return recon_x, latent_vec

  def generate_fmri_from_latent_vec(self, z):
    return self.decoder(z)

def model(latent_dim, fmri_dim):
  return AE(latent_dim=latent_dim, fmri_dim=fmri_dim)
