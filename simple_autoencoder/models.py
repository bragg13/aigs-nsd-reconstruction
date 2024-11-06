# Copyright 2024 The Flax Authors.
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

"""VAE model definitions."""

from flax import linen as nn
from jax import random
import jax.numpy as jnp


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
    z = nn.Dense(224*224*3, name='fc2')(z)
    return z


class AE(nn.Module):
  """Full AE model."""

  latents: int = 20

  def setup(self):
    self.encoder = Encoder(self.latents)
    self.decoder = Decoder()

  def __call__(self, x, z_rng):
    # images, lh_fmri, rh_fmri = x
    # x = jnp.concatenate([images, lh_fmri, rh_fmri], axis=-1)
    latent_vec = self.encoder(x)
    recon_x = self.decoder(latent_vec)
    return recon_x

  def generate(self, z):
    return nn.sigmoid(self.decoder(z))

def model(latents):
  return AE(latents=latents)
