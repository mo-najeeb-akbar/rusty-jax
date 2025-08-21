import flax.linen as nn
import jax.numpy as jnp
from jax import image, random
from functools import partial

class ResidualBlock(nn.Module):
    filters: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        norm = partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, dtype=jnp.float32)
        
        skip = x
        
        # First conv
        x = nn.Conv(self.filters, (3, 3), padding=1, use_bias=False, name='conv1')(x)
        x = norm(name=f'bn1')(x, use_running_average=not train)
        x = nn.relu(x)
        
        # Second conv
        x = nn.Conv(self.filters, (3, 3), padding=1, use_bias=False, name=f'conv2')(x)
        
        # Skip connection
        if skip.shape[-1] != self.filters:
            skip = nn.Conv(self.filters, (1, 1), padding=0, use_bias=False, name=f'skip_conv')(skip)
        
        return nn.relu(x + skip)

class Encoder(nn.Module):
    latent_dim: int
    features: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        norm = partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, dtype=jnp.float32)
        
        # Downsampling blocks
        for i in range(4):
            x = nn.Conv(self.features, (3, 3), strides=(2, 2), padding=1, use_bias=False,
                       name=f'conv_layers.{i}')(x)
            x = norm(name=f'bn_layers.{i}')(x, use_running_average=not train)
            x = nn.relu(x)
            x = ResidualBlock(self.features)(x, train=train)
        
        # Flatten and process - let JAX calculate the actual flattened size
        batch_size = x.shape[0]
        flattened_size = x.shape[1] * x.shape[2] * x.shape[3]  # H * W * C
        # NOTE: need this for the torch
        x = jnp.transpose(x, (0, 3, 1, 2))
        x = x.reshape(batch_size, flattened_size)
        
        x = nn.Dense(128, use_bias=True, name='dense_0')(x)
        x = nn.relu(x)
        
        # VAE outputs
        mu = nn.Dense(self.latent_dim, name='dense_mu')(x)
        log_var = nn.Dense(self.latent_dim, name='dense_logvar')(x)
        
        return mu, log_var

class Decoder(nn.Module):
    latent_dim: int
    bottle_neck: int
    features: int 

    @nn.compact
    def __call__(self, x, train: bool = True):
        norm = partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, dtype=jnp.float32)
        # Initial processing
        x = nn.Dense(128, use_bias=True, name='dense_0')(x)
        x = nn.relu(x)
        
        # Reshape to spatial
        block_size = self.bottle_neck**2 * self.features
        x = nn.Dense(block_size, use_bias=True, name='dense_1')(x)
        x = nn.relu(x)
        x = x.reshape(x.shape[0], self.features, self.bottle_neck, self.bottle_neck)
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Upsampling blocks
        for i in range(4):
            # Resize
            b, h, w, c = x.shape
            x = image.resize(x, (b, h*2, w*2, c), method="bilinear")
            
            # Conv and residual
            x = nn.Conv(self.features, (3, 3), padding=1, use_bias=False, name=f'conv_layers.{i}')(x)
            x = norm(name=f'bn_layers.{i}')(x, use_running_average=not train)
            x = nn.relu(x)
            x = ResidualBlock(self.features)(x, train=train)
        
        # Final output
        x = nn.Conv(1, (3, 3), padding=1, name='out_conv')(x)
        x = nn.sigmoid(x)
        return x

class VAE(nn.Module):
    latent_dim: int = 128
    base_features: int = 48
    block_size: int = 16

    def setup(self):
        self.Encoder = Encoder(self.latent_dim, self.base_features)
        self.Decoder = Decoder(self.latent_dim, self.block_size, self.base_features)
    
    def encode(self, x: jnp.ndarray, training: bool = True): 
        return self.Encoder(x, training)
    
    def decode(self, z: jnp.ndarray, training: bool = True):
        return self.Decoder(z, training)
    
    def reparameterize(self, key: jnp.ndarray, mu: jnp.ndarray, log_var: jnp.ndarray):
        std = jnp.exp(0.5 * log_var)
        eps = random.normal(key, mu.shape)
        return mu + std
    
    def __call__(self, x: jnp.ndarray, key: jnp.ndarray, training: bool = True):
        mu, log_var = self.encode(x, training)
        z = self.reparameterize(key, mu, log_var)
        x_recon = self.decode(z, training)
        return x_recon, mu, log_var
