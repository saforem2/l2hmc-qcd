import random

keys = [
    'global_tf',
    'global_np',
    'x_reverse_check',
    'v_reverse_check',
    'z',
    'vf_init',
    'vb_init',
    'mask_f',
    'mask_a',
    'inference_np',
    'inference_tf',
]

xnet_keys = [
    'dropout',
    'x_layer',
    'v_layer',
    't_layer',
    'h_layer',
    'scale_layer',
    'translation_layer',
    'transformation_layer',
]

vnet_keys = [
    'dropout',
    'x_layer',
    'v_layer',
    't_layer',
    'h_layer',
    'scale_layer',
    'translation_layer',
    'transformation_layer',
]


num_keys = len(keys) + len(xnet_keys) + len(vnet_keys)
nums = random.sample(range(1, int(1e5)), num_keys)

n1 = len(keys)
n2 = len(xnet_keys)
ints, net_ints = nums[:n1], nums[n1:]
xints, vints = net_ints[:n2], net_ints[n2:]

seeds = dict(zip(keys, ints))
xnet_seeds = dict(zip(xnet_keys, xints))
vnet_seeds = dict(zip(vnet_keys, vints))
