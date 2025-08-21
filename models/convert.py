import torch
import jax.numpy as jnp
import numpy as np
import jax.random as random
from flax import linen as nn
from jax_vae import VAE as JVAE
from pytorch_vae import VAE as TVAE
from functools import reduce
from operator import getitem
import torch.onnx

get_nested = lambda d, keys: reduce(getitem, keys, d)


def fuse_conv_bn(model):
    """Fuse Conv2d and BatchNorm2d layers"""
    model = torch.jit.script(model)
    model = torch.jit.optimize_for_inference(model)
    return model

def convert_bn(params_j, batch_stats_j, params_t, keys_j, key_t):
    bn_ = get_nested(params_j, keys_j)
    bias = np.array(bn_['bias'])
    scale = np.array(bn_['scale'])
    bn_ = get_nested(batch_stats_j, keys_j)
    mean = np.array(bn_['mean'])
    var = np.array(bn_['var'])

    params_t[f'{key_t}.weight'].copy_(torch.from_numpy(scale))
    params_t[f'{key_t}.bias'].copy_(torch.from_numpy(bias))
    params_t[f'{key_t}.running_mean'].copy_(torch.from_numpy(mean))
    params_t[f'{key_t}.running_var'].copy_(torch.from_numpy(var))
    
    print('happy, converted', key_t)

def convert_conv(params_j, batch_stats_j, params_t, keys_j, key_t):
    conv = get_nested(params_j, keys_j)
    conv_kernel = np.transpose(np.array(conv['kernel']), (3, 2, 0, 1))
    params_t[f'{key_t}.weight'].copy_(torch.from_numpy(conv_kernel))

    if 'bias' in conv.keys():
        bias = np.array(conv['bias'])
        params_t[f'{key_t}.bias'].copy_(torch.from_numpy(bias))
    
    print('happy, converted', key_t)

def convert_dense(params_j, batch_stats_j, params_t, keys_j, key_t):
    dense = get_nested(params_j, keys_j)
    dense_kernel = np.transpose(np.array(dense['kernel']), (1, 0))
    params_t[f'{key_t}.weight'].copy_(torch.from_numpy(dense_kernel))
    
    if 'bias' in dense.keys():
        bias = np.array(dense['bias'])
        params_t[f'{key_t}.bias'].copy_(torch.from_numpy(bias))

    print('happy, converted', key_t)


def router(name):
    if 'bn' in name:
        return convert_bn
    elif 'conv' in name:
        return convert_conv
    elif 'dense' in name:
        return convert_dense

def pretty_print_dict(data, indent=0, path=[], path_torch=[], params_j=None, batch_stats_j=None, params_t=None):
    spacing = "  " * indent

    for key, value in data.items():
        if 'Residual' in key:
            num = key.split('_')[-1]
            key_torch = 'residual_blocks.' + num
        else:
            key_torch = key.lower()
        current_path = path + [key]  # Create new path for this key
        current_path_torch = path_torch + [key_torch]
        
        if isinstance(key, str) and key[0].isupper():
            # This is a subdictionary
            if isinstance(value, dict):
                pretty_print_dict(value, indent + 1, current_path, current_path_torch, params_j, batch_stats_j, params_t)
        else:
            # Regular key-value pair
            # print(f"{spacing}{key}: (path: {' -> '.join(current_path)})")
            # tmp = get_nested(batch_stats_j, current_path)
            # print(value.keys())
            final_torch_key = '.'.join(current_path_torch)
            # print(final_torch_key)
            router(key)(params_j, batch_stats_j, params_t, current_path, final_torch_key) 

jvae = JVAE()

key = random.key(0)
key, *subkeys = random.split(key, 3)
x = random.normal(subkeys[0], (1, 256, 256, 1))
x_torch = torch.tensor(np.array(x)).permute(0, 3, 1, 2)

params = jvae.init(subkeys[1], x, subkeys[1], training=True)

# print(nn.tabulate(JVAE(), random.key(0))(x, random.key(0), training=False))

tvae = TVAE()
tvae.eval()
params_torch = tvae.state_dict()

# print(params_torch.keys())
pretty_print_dict(params['params'], params_j=params['params'], batch_stats_j=params['batch_stats'], params_t=params_torch)

reconj, muj, logvarj = jvae.apply(params, x, key, training=False)

with torch.no_grad():
    recont, mut, logvart = tvae(x_torch)

np.testing.assert_almost_equal(np.array(reconj), recont.permute(0, 2, 3, 1).numpy(), decimal=3)
np.testing.assert_almost_equal(np.array(logvarj), logvart.numpy(), decimal=3)
np.testing.assert_almost_equal(np.array(muj), mut.numpy(), decimal=3)

dummy_input = torch.randn(1, 1, 256, 256)
torch.onnx.export(
    tvae,                          # PyTorch model
    dummy_input,                    # Model input
    "vae.onnx",                   # Output file name
    opset_version=16,               # ONNX version 16+ for Burn
    do_constant_folding=True,       # Optimize constant folding
    input_names=['input'],          # Input names
    output_names=['output'],        # Output names
    optimize=True,                    # Enable ONNX optimizer
)

import onnxsim
import onnx
model_simplified, check = onnxsim.simplify("vae.onnx")
onnx.save(model_simplified, "vae.onnx")
print(f"Simplification successful: {check}")
