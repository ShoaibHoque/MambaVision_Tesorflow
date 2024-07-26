#!/usr/bin/env python3

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import tensorflow as tf
import tensorflow_addons as tfa
import math
from pathlib import Path
import numpy as np
import h5py
import logging
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import TruncatedNormal, Constant
from tensorflow.keras.layers.experimental import SyncBatchNormalization as BatchNormalization

def _cfg(url='', **kwargs):
    return {'url': url,
            'num_classes': 1000,
            'input_size': (3, 224, 224),
            'pool_size': None,
            'crop_pct': 0.875,
            'interpolation': 'bicubic',
            'fixed_input_size': True,
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            **kwargs
            }

default_cfgs = {
    'mamba_vision_T': _cfg(url='https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_T2': _cfg(url='https://huggingface.co/nvidia/MambaVision-T2-1K/resolve/main/mambavision_tiny2_1k.pth.tar',
                            crop_pct=0.98,
                            input_size=(3, 224, 224),
                            crop_mode='center'),
    'mamba_vision_S': _cfg(url='https://huggingface.co/nvidia/MambaVision-S-1K/resolve/main/mambavision_small_1k.pth.tar',
                           crop_pct=0.93,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_B': _cfg(url='https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_L': _cfg(url='https://huggingface.co/nvidia/MambaVision-L-1K/resolve/main/mambavision_large_1k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_L2': _cfg(url='https://huggingface.co/nvidia/MambaVision-L2-1K/resolve/main/mambavision_large2_1k.pth.tar',
                            crop_pct=1.0,
                            input_size=(3, 224, 224),
                            crop_mode='center')                                
}

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size: window size
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, H, W, C = x.shape
    x = tf.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    windows = tf.reshape(tf.transpose(x, perm=[0, 1, 3, 2, 4, 5]), (-1, window_size*window_size, C))
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size*window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = tf.reshape(windows, (B, H // window_size, W // window_size, window_size, window_size, -1))
    x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 4, 3, 5]), (B, H, W, windows.shape[2]))
    return x

def _load_state_dict(module, state_dict, strict=False, logger=None):
    """
    Load state_dict to a module.

    This method is modified from :meth:`tf.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~tf.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    
    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys, err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None
    missing_keys = [key for key in all_missing_keys if 'num_batches_tracked' not in key]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)

def _load_state_dict(model, state_dict, strict=False, logger=None):
    """
    Load state_dict into model.

    Args:
        model (tf.keras.Model): The model to load weights into.
        state_dict (dict): The state dictionary containing model weights.
        strict (bool): Whether to enforce that the keys in state_dict match the model's keys exactly.
        logger (logging.Logger or None): The logger for error messages.

    Returns:
        None
    """
    model_dict = {layer.name: layer for layer in model.layers}
    for key, value in state_dict.items():
        if key in model_dict:
            layer = model_dict[key]
            if isinstance(layer, layers.Dense):
                layer.kernel.assign(value['kernel'])
                if 'bias' in value:
                    layer.bias.assign(value['bias'])
            elif isinstance(layer, layers.Conv2D):
                layer.kernel.assign(value['kernel'])
                if 'bias' in value:
                    layer.bias.assign(value['bias'])
            elif isinstance(layer, layers.BatchNormalization):
                layer.gamma.assign(value['gamma'])
                layer.beta.assign(value['beta'])
                layer.moving_mean.assign(value['moving_mean'])
                layer.moving_variance.assign(value['moving_variance'])
            else:
                if logger:
                    logger.warning(f"Layer {key} is not recognized for custom loading.")
        elif strict:
            raise ValueError(f"Key {key} not found in model layers.")
        else:
            if logger:
                logger.warning(f"Key {key} not found in model layers. Ignoring.")

def _load_checkpoint(model, filename, strict=False, logger=None):
    """
    Load checkpoint from a file.

    Args:
        model (tf.keras.Model): Model to load checkpoint.
        filename (str): Path to the checkpoint file.
        strict (bool): Whether to enforce that the keys in state_dict match the model's keys exactly.
        logger (logging.Logger or None): The logger for error messages.

    Returns:
        dict: The loaded checkpoint.
    """
    checkpoint = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            checkpoint[key] = {k: np.array(f[key][k]) for k in f[key].keys()}

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Adjust for different prefixes or formats if needed
    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class Downsample(layers.Layer):
    """
    Down-sampling block"
    """

    def __init__(self, dim, keep_dim=False):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """
        super(Downsample, self).__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = tf.keras.Sequential([
            layers.Conv2D(dim_out, kernel_size=3, strides=2, padding='same', use_bias=False)
        ])

    def call(self, x):
        x = self.reduction(x)
        return x

class PatchEmbed(layers.Layer):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        super(PatchEmbed, self).__init__()
        self.proj = layers.Layer()
        self.conv_down = tf.keras.Sequential([
            layers.Conv2D(in_dim, kernel_size=3, strides=2, padding='same', use_bias=False),
            tfa.layers.GroupNormalization(groups=1, axis=-1, epsilon=1e-4),
            layers.ReLU(),
            layers.Conv2D(dim, kernel_size=3, strides=2, padding='same', use_bias=False),
            tfa.layers.GroupNormalization(groups=1, axis=-1, epsilon=1e-4),
            layers.ReLU()
        ])

    def call(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x

class ConvBlock(layers.Layer):

    def __init__(self, dim, drop_path=0., layer_scale=None, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv1 = layers.Conv2D(dim, dim, kernel_size=kernel_size, strides=1, padding='same')
        self.norm1 = layers.BatchNormalization(epsilon=1e-5)
        self.act1 = layers.Activation('gelu')
        self.conv2 = layers.Conv2D(dim, dim, kernel_size=kernel_size, strides=1, padding='same')
        self.norm2 = layers.BatchNormalization(epsilon=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and isinstance(layer_scale, (int, float)):
            self.gamma = tf.Variable(layer_scale * tf.ones((dim,)), trainable=True)
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = tfa.layers.StochasticDepth(drop_path) if drop_path > 0. else tf.keras.layers.Layer()

    def call(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * tf.reshape(self.gamma, (1, 1, 1, -1))
        x = input + self.drop_path(x)
        return x


class MambaVisionMixer(layers.Layer):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
    ):
        super(MambaVisionMixer, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = layers.Dense(self.d_inner, use_bias=bias)    
        self.x_proj = layers.Dense(self.dt_rank + self.d_state * 2, use_bias=False)
        self.dt_proj = layers.Dense(self.d_inner // 2, use_bias=True)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            self.dt_proj.kernel_initializer = tf.keras.initializers.Constant(dt_init_std)
        elif dt_init == "random":
            self.dt_proj.kernel_initializer = tf.keras.initializers.RandomUniform(-dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = tf.exp(
            tf.random.uniform((self.d_inner // 2,), minval=math.log(dt_min), maxval=math.log(dt_max))
        ).numpy().clip(min=dt_init_floor)
        inv_dt = dt + tf.math.log(-tf.math.expm1(-dt))
        self.dt_proj.bias.assign(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = tf.tile(
            tf.range(1, self.d_state + 1, dtype=tf.float32)[tf.newaxis, :],
            [self.d_inner // 2, 1]
        )
        self.A_log = tf.Variable(tf.math.log(A), trainable=True)
        self.A_log._no_weight_decay = True
        self.D = tf.Variable(tf.ones((self.d_inner // 2,)), trainable=True)
        self.D._no_weight_decay = True
        self.out_proj = layers.Dense(self.d_model, use_bias=bias)
        self.conv1d_x = layers.Conv1D(
            self.d_inner // 2, kernel_size=d_conv, use_bias=conv_bias, groups=self.d_inner // 2
        )
        self.conv1d_z = layers.Conv1D(
            self.d_inner // 2, kernel_size=d_conv, use_bias=conv_bias, groups=self.d_inner // 2
        )

    def call(self, hidden_states):
        B, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = tf.transpose(xz, perm=[0, 2, 1])
        x, z = tf.split(xz, 2, axis=1)
        A = -tf.exp(self.A_log)
        x = tfa.activations.gelu(self.conv1d_x(x))
        z = tfa.activations.gelu(self.conv1d_z(z))
        x_dbl = self.x_proj(tf.reshape(x, (-1, x.shape[-1])))
        dt, B, C = tf.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], axis=-1)
        dt = tf.reshape(self.dt_proj(dt), (B, self.d_inner // 2, seqlen))
        B = tf.reshape(B, (B, self.d_state, seqlen))
        C = tf.reshape(C, (B, self.d_state, seqlen))
        y = selective_scan_fn(x, dt, A, B, C, self.D, z=None, delta_bias=self.dt_proj.bias, delta_softplus=True)
        
        y = tf.concat([y, z], axis=1)
        y = tf.transpose(y, perm=[0, 2, 1])
        out = self.out_proj(y)
        return out


class Attention(layers.Layer):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=layers.LayerNormalization,
    ):
        super(Attention, self).__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.q_norm = norm_layer(axis=-1) if qk_norm else tf.keras.layers.Layer()
        self.k_norm = norm_layer(axis=-1) if qk_norm else tf.keras.layers.Layer()
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)

    def call(self, x):
        B, N, C = x.shape
        qkv = tf.reshape(self.qkv(x), (B, N, 3, self.num_heads, self.head_dim))
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = tf.unstack(qkv, axis=0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = layers.Attention(dropout=self.attn_drop.rate)([q, k, v])
        else:
            q = q * self.scale
            attn = tf.matmul(q, k, transpose_b=True)
            attn = tf.nn.softmax(attn, axis=-1)
            attn = self.attn_drop(attn)
            x = tf.matmul(attn, v)

        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(layers.Layer):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 counter, 
                 transformer_blocks, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=False, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=layers.Activation('gelu'), 
                 norm_layer=layers.LayerNormalization, 
                 Mlp_block=layers.Dense,
                 layer_scale=None,
                 ):
        super(Block, self).__init__()
        self.norm1 = norm_layer(axis=-1)
        if counter in transformer_blocks:
            self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        else:
            self.mixer = MambaVisionMixer(d_model=dim, 
                                          d_state=8,  
                                          d_conv=3,    
                                          expand=1
                                          )

        self.drop_path = tfa.layers.StochasticDepth(drop_path) if drop_path > 0. else tf.keras.layers.Layer()
        self.norm2 = norm_layer(axis=-1)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(dim, mlp_hidden_dim, activation=act_layer, use_bias=False)
        use_layer_scale = layer_scale is not None and isinstance(layer_scale, (int, float))
        self.gamma_1 = tf.Variable(layer_scale * tf.ones((dim,)), trainable=True) if use_layer_scale else 1
        self.gamma_2 = tf.Variable(layer_scale * tf.ones((dim,)), trainable=True) if use_layer_scale else 1

    def call(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(layers.Layer):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks = [],
    ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super(MambaVisionLayer, self).__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = [ConvBlock(dim=dim,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     layer_scale=layer_scale_conv)
                           for i in range(depth)]
            self.transformer_block = False
        else:
            self.transformer_block = True
            self.blocks = [Block(dim=dim,
                                 counter=i, 
                                 transformer_blocks=transformer_blocks,
                                 num_heads=num_heads,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 layer_scale=layer_scale)
                           for i in range(depth)]
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        self.window_size = window_size

    def call(self, x):
        _, _, H, W = x.shape

        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = tf.pad(x, [[0, 0], [0, 0], [0, pad_r], [0, pad_b]])
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)

        for blk in self.blocks:
            x = blk(x)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W]
        if self.downsample is None:
            return x
        return self.downsample(x)


class MambaVision(Model):
    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):
        super(MambaVision, self).__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = tf.linspace(0.0, drop_path_rate, sum(depths))
        self.levels = []
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            level = MambaVisionLayer(
                dim=int(dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                conv=conv,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=(i < 3),
                layer_scale=layer_scale,
                layer_scale_conv=layer_scale_conv,
                transformer_blocks=list(range(depths[i] // 2 + 1, depths[i])) if depths[i] % 2 != 0 else list(range(depths[i] // 2, depths[i]))
            )
            self.levels.append(level)
        self.norm = layers.BatchNormalization()
        self.avgpool = layers.GlobalAveragePooling2D()
        self.head = layers.Dense(num_features, activation='linear') if num_classes > 0 else tf.keras.layers.Layer()
        self.build((None, None, None, in_chans))
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, layers.Dense):
            layer.kernel_initializer = TruncatedNormal(stddev=0.02)
            if layer.bias is not None:
                layer.bias_initializer = Constant(0)
        elif isinstance(layer, layers.LayerNormalization) or isinstance(layer, BatchNormalization):
            layer.gamma_initializer = Constant(1.0)
            layer.beta_initializer = Constant(0.0)

    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        for level in self.levels:
            x = level(x)
        x = self.norm(x)
        x = self.avgpool(x)
        return x

    def call(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def load_state_dict(self, pretrained, strict=False):
        # Implement the load state dict logic
        pass

def register_model(fn):
    # Assuming a registration function decorator is available
    return fn

@register_model
def mamba_vision_T(pretrained=False, **kwargs):
    model = MambaVision(
        depths=[1, 3, 8, 4],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 14, 7],
        dim=80,
        in_dim=32,
        mlp_ratio=4,
        drop_path_rate=0.2,
        **kwargs)
    if pretrained:
        model_path = kwargs.pop("model_path", "/tmp/mamba_vision_T.pth.tar")
        # Load pretrained weights
    return model

@register_model
def mamba_vision_T2(pretrained=False, **kwargs):
    model = MambaVision(
        depths=[1, 3, 11, 4],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 14, 7],
        dim=80,
        in_dim=32,
        mlp_ratio=4,
        drop_path_rate=0.2,
        **kwargs)
    if pretrained:
        model_path = kwargs.pop("model_path", "/tmp/mamba_vision_T2.pth.tar")
        # Load pretrained weights
    return model

@register_model
def mamba_vision_S(pretrained=False, **kwargs):
    model = MambaVision(
        depths=[3, 3, 7, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 14, 7],
        dim=96,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.2,
        **kwargs)
    if pretrained:
        model_path = kwargs.pop("model_path", "/tmp/mamba_vision_S.pth.tar")
        # Load pretrained weights
    return model

@register_model
def mamba_vision_B(pretrained=False, **kwargs):
    model = MambaVision(
        depths=[3, 3, 10, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 14, 7],
        dim=128,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.3,
        layer_scale=1e-5,
        layer_scale_conv=None,
        **kwargs)
    if pretrained:
        model_path = kwargs.pop("model_path", "/tmp/mamba_vision_B.pth.tar")
        # Load pretrained weights
    return model

@register_model
def mamba_vision_L(pretrained=False, **kwargs):
    model = MambaVision(
        depths=[3, 3, 10, 5],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8, 14, 7],
        dim=196,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.3,
        layer_scale=1e-5,
        layer_scale_conv=None,
        **kwargs)
    if pretrained:
        model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L.pth.tar")
        # Load pretrained weights
    return model

@register_model
def mamba_vision_L2(pretrained=False, **kwargs):
    model = MambaVision(
        depths=[3, 3, 12, 5],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8, 14, 7],
        dim=196,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.3,
        layer_scale=1e-5,
        layer_scale_conv=None,
        **kwargs)
    if pretrained:
        model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L2.pth.tar")
        # Load pretrained weights
    return model
