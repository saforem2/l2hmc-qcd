"""
network.py

Implements network architecture used to train L2HMC algorithm on 2D U(1)
lattice gauge model.

Author: Sam Foreman (github: @saforem2)
Date: 07/22/2019
"""
from __future__ import print_function, division, absolute_import
import pickle

import numpy as np
import tensorflow as tf

from .conv_net import ConvNet2D, ConvNet3D
from .generic_net import GenericNet
from config import Weights

#  import utils.file_io as io
from seed_dict import seeds, vnet_seeds, xnet_seeds

#  from config import GLOBAL_SEED


np.random.seed(seeds['global_np'])

if '2.' not in tf.__version__:
    tf.set_random_seed(seeds['global_tf'])


class BaseNet(tf.keras.Model):
    """Base class for building networks."""
    def __init__(self, model_name=None, **kwargs):
        super(BaseNet, self).__init__(name=model_name)

    def get_layer_weights(self, sess):
        """Extract numerical values of all layer weights."""
        weights_dict = {}
        for name, layer in self.layers_dict.items():
            weights_dict[name] = {}
            if isinstance(layer, dict):
                for subname, sublayer in layer.items():
                    w, b = sess.run(sublayer.weights)
                    weights_dict[name][subname] = Weights(w=w, b=b)
            else:
                w, b = sess.run(layer.weights)
                weights_dict[name] = Weights(w=w, b=b)

        return weights_dict

    def save_layer_weights(self, sess, out_file):
        weights_dict = self.get_layer_weights(sess)
        with open(out_file, 'wb') as f:
            pickle.dump(weights_dict, f)

        fpath, ext = out_file.split('.')
        types_file = f'{fpath}_types.{ext}'
        with open(types_file, 'wb') as f:
            pickle.dump(self.layers_types, types_file)

        return weights_dict

    def call(self, inputs, train_phase):
        raise NotImplementedError


class FullNet(tf.keras.Model):
    """Complete network used for training L2HMC model."""
    def __init__(self, model_name, **kwargs):
        """
        Initialization method.

        Args:
            model_name: Name of the model.
            kwargs: Keyword arguments used to specify specifics of
                convolutional structure.
        """
        super(FullNet, self).__init__(name=model_name)

        if model_name == 'XNet':
            generic_name_scope = 'GenericNetX'
            kwargs['net_seeds'] = xnet_seeds

        elif model_name == 'VNet':
            generic_name_scope = 'GenericNetV'
            kwargs['net_seeds'] = vnet_seeds

        self.layers_dict = {}
        with tf.name_scope(model_name):
            kwargs['net_name'] = 'ConvNetX'
            network_arch = kwargs.get('network_arch', 'conv3D')

            if network_arch == 'conv2D':
                self.x_conv_net = ConvNet2D('ConvNet2Dx', **kwargs)

                kwargs['net_name'] = 'ConvNetV'
                self.v_conv_net = ConvNet2D('ConvNet2Dv', **kwargs)

            elif network_arch == 'conv3D':
                self.x_conv_net = ConvNet3D('ConvNet3Dx', **kwargs)

                kwargs['net_name'] = 'ConvNetV'
                self.v_conv_net = ConvNet3D('ConvNet3Dv', **kwargs)

            else:
                self.x_conv_net = None
                self.v_conv_net = None

            kwargs['net_name'] = generic_name_scope
            self.generic_net = GenericNet("GenericNet", **kwargs)

        self.layers_dict.update(**self.generic_net.layers_dict)

        if self.x_conv_net is not None:
            self.layers_dict.update(**self.x_conv_net.layers_dict)
            #  self.layers_dict.update(**self.x_conv_net.layers_dict)
            self.layers_dict.update(**self.v_conv_net.layers_dict)

    def get_layer_weights(self, sess):
        """Extract numerical values of all layer weights."""
        weights_dict = self.generic_net.get_weights(sess)

        if self.x_conv_net is not None:
            weights_dict.update(**self.x_conv_net.get_layer_weights(sess))
            weights_dict.update(**self.v_conv_net.get_layer_weights(sess))

        return weights_dict

    def save_layer_weights(self, sess, out_file):
        """Save all layer weights to `out_file`."""
        weights_dict = self.get_layer_weights(sess)
        with open(out_file, 'wb') as f:
            pickle.dump(weights_dict, f)

        return weights_dict

    def call(self, inputs, train_phase):
        """Call the network (forward pass)."""
        v, x, t = inputs

        if self.x_conv_net is not None:
            v = self.v_conv_net(v, train_phase)
            x = self.x_conv_net(x, train_phase)

        scale, translation, transformation = self.generic_net([v, x, t],
                                                              train_phase)

        return scale, translation, transformation
