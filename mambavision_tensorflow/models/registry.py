import tensorflow as tf
import numpy as np
import os
import re
import fnmatch
from collections import defaultdict, OrderedDict
from copy import deepcopy
import sys

__all__ = ['list_models', 'is_model', 'model_entrypoint', 'list_modules', 'is_model_in_modules',
           'is_model_default_key', 'has_model_default_key', 'get_model_default_value', 'is_model_pretrained']

_module_to_models = defaultdict(set)
_model_to_module = {}
_model_entrypoints = {}
_model_has_pretrained = set()
_model_default_cfgs = dict()

def register_pip_model(fn):
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)

    has_pretrained = False
    if hasattr(mod, 'default_cfgs') and model_name in mod.default_cfgs:
        has_pretrained = 'url' in mod.default_cfgs[model_name] and 'http' in mod.default_cfgs[model_name]['url']
        _model_default_cfgs[model_name] = deepcopy(mod.default_cfgs[model_name])
    if has_pretrained:
        _model_has_pretrained.add(model_name)
    return fn

def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

def list_models(filter='', module='', pretrained=False, exclude_filters='', name_matches_cfg=False):
    if module:
        all_models = list(_module_to_models[module])
    else:
        all_models = _model_entrypoints.keys()

    if filter:
        models = []
        include_filters = filter if isinstance(filter, (tuple, list)) else [filter]
        for f in include_filters:
            include_models = fnmatch.filter(all_models, f)
            if len(include_models):
                models = set(models).union(include_models)
    else:
        models = all_models

    if exclude_filters:
        if not isinstance(exclude_filters, (tuple, list)):
            exclude_filters = [exclude_filters]
        for xf in exclude_filters:
            exclude_models = fnmatch.filter(models, xf)
            if len(exclude_models):
                models = set(models).difference(exclude_models)

    if pretrained:
        models = _model_has_pretrained.intersection(models)

    if name_matches_cfg:
        models = set(_model_default_cfgs).intersection(models)

    return list(sorted(models, key=_natural_key))

def is_model(model_name):
    return model_name in _model_entrypoints

def model_entrypoint(model_name):
    return _model_entrypoints[model_name]

def list_modules():
    modules = _module_to_models.keys()
    return list(sorted(modules))

def is_model_in_modules(model_name, module_names):
    assert isinstance(module_names, (tuple, list, set))
    return any(model_name in _module_to_models[n] for n in module_names)

def has_model_default_key(model_name, cfg_key):
    if model_name in _model_default_cfgs and cfg_key in _model_default_cfgs[model_name]:
        return True
    return False

def is_model_default_key(model_name, cfg_key):
    if model_name in _model_default_cfgs and _model_default_cfgs[model_name].get(cfg_key, False):
        return True
    return False

def get_model_default_value(model_name, cfg_key):
    if model_name in _model_default_cfgs:
        return _model_default_cfgs[model_name].get(cfg_key, None)
    else:
        return None

def is_model_pretrained(model_name):
    return model_name in _model_has_pretrained

def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = np.load(checkpoint_path, allow_pickle=True).item()
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            state_dict = checkpoint[state_dict_key]
        else:
            state_dict = checkpoint
        print("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        print("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    state_dict = load_state_dict(checkpoint_path, use_ema)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Layer):
            if layer.name in state_dict:
                weights = state_dict[layer.name]
                layer.set_weights(weights)
            elif strict:
                raise ValueError(f"Layer {layer.name} not found in checkpoint.")
            else:
                print(f"Layer {layer.name} not found in checkpoint. Skipping.")

def create_model(
        model_name,
        pretrained=False,
        checkpoint_path='',
        **kwargs):
    create_fn = model_entrypoint(model_name)
    model = create_fn(pretrained=pretrained, **kwargs)
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model
