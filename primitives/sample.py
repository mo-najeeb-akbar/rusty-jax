import numpy as np
import torch
import jax.numpy as jnp
from jax import random
import flax.linen as nn


def compare_fc():
    t_conv = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2, padding='valid')

    kernel = t_conv.weight.detach().cpu().numpy()
    bias = t_conv.bias.detach().cpu().numpy()

    # [outC, inC, kH, kW] -> [kH, kW, inC, outC]
    kernel = jnp.transpose(kernel, (2, 3, 1, 0))

    key = random.key(0)
    x = random.normal(key, (1, 6, 6, 3))

    variables = {'params': {'kernel': kernel, 'bias': bias}}
    j_conv = nn.Conv(features=4, kernel_size=(2, 2), padding='VALID')
    j_out = j_conv.apply(variables, x)

    # [N, H, W, C] -> [N, C, H, W]
    t_x = torch.from_numpy(np.transpose(np.array(x), (0, 3, 1, 2)))
    t_out = t_conv(t_x)
    # [N, C, H, W] -> [N, H, W, C]
    t_out = np.transpose(t_out.detach().cpu().numpy(), (0, 2, 3, 1))

    np.testing.assert_almost_equal(j_out, t_out, decimal=6)
    print("✅ Convolutions are equivalent!")

def compare_conv():
    # kernel size - 1 // stride
    padding = (3 - 1) // 2 
    
    t_conv = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=padding)

    kernel = t_conv.weight.detach().cpu().numpy()
    bias = t_conv.bias.detach().cpu().numpy()

    # [outC, inC, kH, kW] -> [kH, kW, inC, outC]
    kernel = jnp.transpose(kernel, (2, 3, 1, 0))

    key = random.key(0)
    x = random.normal(key, (1, 6, 6, 3))

    variables = {'params': {'kernel': kernel, 'bias': bias}}
    j_conv = nn.Conv(features=4, kernel_size=(3, 3), strides=2, padding=padding)
    j_out = j_conv.apply(variables, x)

    # [N, H, W, C] -> [N, C, H, W]
    t_x = torch.from_numpy(np.transpose(np.array(x), (0, 3, 1, 2)))
    t_out = t_conv(t_x)
    # [N, C, H, W] -> [N, H, W, C]
    t_out = np.transpose(t_out.detach().cpu().numpy(), (0, 2, 3, 1))
    
    print(t_out.shape)
    np.testing.assert_almost_equal(j_out, t_out, decimal=6)
    print('happy')


def compare_conv_fc():
    class TModel(torch.nn.Module):

        def __init__(self):
            super(TModel, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels=3, out_channels=4, stride=2, kernel_size=3, padding=1)
            self.fc = torch.nn.Linear(in_features=36, out_features=2)

        def forward(self, x):
            x = self.conv(x)
            x = x.reshape(x.shape[0], -1)
            x = self.fc(x)
            return x
    class JModel(nn.Module):

        @nn.compact
        def __call__(self, x):
            x = nn.Conv(features=4, kernel_size=(3, 3), strides=2, padding=1, name='conv')(x)
            x = jnp.transpose(x, (0, 3, 1, 2))
            x = jnp.reshape(x, (x.shape[0], -1))
            x = nn.Dense(features=2, name='fc')(x)
            return x

    model = JModel()
    key1, key2 = random.split(random.key(0))
    x_jax = random.normal(key1, (1, 6, 6, 3))  # Dummy input data (NHWC format)
    params = model.init(key2, x_jax)

    # Initialize PyTorch model
    tnet = TModel()
    tnet.eval()
    params_torch = tnet.state_dict()

    conv_kernel = np.transpose(np.array(params['params']['conv']['kernel']), (3, 2, 0, 1))
    conv_bias = np.array(params['params']['conv']['bias'])
    
    # FC: Flax (C_in, C_out) → PyTorch (C_out, C_in)
    fc_kernel = np.transpose(np.array(params['params']['fc']['kernel']), (1, 0))
    fc_bias = np.array(params['params']['fc']['bias'])
    # Copy weights to PyTorch model
    params_torch['conv.weight'].copy_(torch.from_numpy(conv_kernel))
    params_torch['conv.bias'].copy_(torch.from_numpy(conv_bias))
    params_torch['fc.weight'].copy_(torch.from_numpy(fc_kernel))
    params_torch['fc.bias'].copy_(torch.from_numpy(fc_bias))
    print(params_torch['fc.weight']) 
    x_torch = torch.from_numpy(np.array(x_jax)).permute(0, 3, 1, 2).float()
    
    # Forward pass
    with torch.no_grad():
        output_torch = tnet(x_torch)
    
    output_jax = model.apply(params, x_jax)
    
    # Compare outputs
    output_torch_np = output_torch.numpy()
    output_jax_np = np.array(output_jax)

    np.testing.assert_allclose(output_torch_np, output_jax_np, rtol=1e-5, atol=1e-6)
    print("✅ Outputs match! Weight conversion successful.")


if __name__ == "__main__":
    compare_fc()
    compare_conv()
    compare_conv_fc()
