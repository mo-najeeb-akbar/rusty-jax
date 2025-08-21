import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, filters, block_name=None):
        super().__init__()
        self.filters = filters
        self.block_name = block_name or "res"
        
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters, momentum=0.1, eps=1e-5)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        
    def forward(self, x):
        skip = x
        
        # First conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second conv
        out = self.conv2(out)
        
        return F.relu(out + skip)

class Encoder(nn.Module):
    def __init__(self, latent_dim, features, input_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.features = features
        
        # Calculate the size after 4 downsampling operations (stride=2 each)
        # input_size -> input_size/2 -> input_size/4 -> input_size/8 -> input_size/16
        self.final_spatial_size = input_size // (2**4)
        self.flattened_size = self.final_spatial_size * self.final_spatial_size * features
        
        # Downsampling layers - use padding=1 to match JAX SAME padding for 3x3 kernels
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1 if i == 0 else features, features, 3, stride=2, padding=1, bias=False)
            for i in range(4)
        ])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(features, momentum=0.1, eps=1e-5) for _ in range(4)
        ])
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(features) for i in range(4)
        ])
        
        # Dense layers - using calculated flattened size
        self.dense_0 = nn.Linear(self.flattened_size, 128, bias=True)
        # self.bn_0 = nn.BatchNorm1d(128, momentum=0.1, eps=1e-5)
        
        # VAE outputs
        self.dense_mu = nn.Linear(128, latent_dim)
        self.dense_logvar = nn.Linear(128, latent_dim)
    
    def forward(self, x):
        # Downsampling blocks
        for i in range(4):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = F.relu(x)
            x = self.residual_blocks[i](x)
        
        # Flatten and process - use reshape instead of view
        x = x.reshape(x.size(0), -1)
        x = self.dense_0(x)
        x = F.relu(x)
        
        # VAE outputs
        mu = self.dense_mu(x)
        log_var = self.dense_logvar(x)
        
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, bottle_neck, features):
        super().__init__()
        self.latent_dim = latent_dim
        self.bottle_neck = bottle_neck
        self.features = features
        
        # Initial processing
        self.dense_0 = nn.Linear(latent_dim, 128, bias=True)
        
        # Reshape layer
        block_size = bottle_neck**2 * features
        self.dense_1 = nn.Linear(128, block_size, bias=True)
        
        # Upsampling layers
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(features, features, 3, padding=1, bias=False) for _ in range(4)
        ])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(features, momentum=0.1, eps=1e-5) for _ in range(4)
        ])
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(features) for i in range(4)
        ])
        
        # Final output
        self.out_conv = nn.Conv2d(features, 1, 3, padding=1)
    
    def forward(self, x):
        # Initial processing
        x = self.dense_0(x)
        x = F.relu(x)
        # Reshape to spatial
        x = self.dense_1(x)
        x = F.relu(x)
        x = x.reshape(x.size(0), self.features, self.bottle_neck, self.bottle_neck)
        
        # Upsampling blocks
        for i in range(4):
            # Resize (bilinear interpolation)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, antialias=False)
            
            # Conv and residual
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = F.relu(x)
            x = self.residual_blocks[i](x)
        
        # Final output
        x = self.out_conv(x)
        x = torch.sigmoid(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=128, base_features=48, block_size=16, input_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_features = base_features
        self.block_size = block_size
        self.input_size = input_size
        
        self.encoder = Encoder(latent_dim, base_features, input_size)
        self.decoder = Decoder(latent_dim, block_size, base_features)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std 
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var
