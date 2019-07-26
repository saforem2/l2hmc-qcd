"""
network.py

Implements network architecture used to train L2HMC algorithm on 2D U(1)
lattice gauge model.

Author: Sam Foreman (github: @saforem2)
Date: 07/22/2019
"""
import numpy as np
import tensorflow as tf
from .conv_net3d import ConvNet3D
from .conv_net2d import ConvNet2D
from .generic_net import GenericNet
import utils.file_io as io
#  import utils.file_io as io

from variables import GLOBAL_SEED


np.random.seed(GLOBAL_SEED)

if '2.' not in tf.__version__:
    tf.set_random_seed(GLOBAL_SEED)


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
            generic_name_scope = 'x_generic_block'
        elif model_name == 'VNet':
            generic_name_scope = 'v_generic_block'

        with tf.name_scope(model_name):
            kwargs['name_scope'] = 'x_conv_block'
            network_arch = kwargs.get('network_arch', 'conv3D')

            if network_arch == 'conv2D':
                io.log('Using ConvNet2D architecture...')
                self.x_conv_net = ConvNet2D('ConvNet2Dx', **kwargs)

                kwargs['name_scope'] = 'v_conv_block'
                self.v_conv_net = ConvNet2D('ConvNet2Dv', **kwargs)

            elif network_arch == 'conv3D':
                io.log('Using ConvNet3D architecture...')
                self.x_conv_net = ConvNet3D('ConvNet3Dx', **kwargs)

                kwargs['name_scope'] = 'v_conv_block'
                self.v_conv_net = ConvNet3D('ConvNet3Dv', **kwargs)

            else:
                io.log('Using GenericNet architecture...')
                self.x_conv_net = self.v_conv_net = None

            #  kwargs['name_scope'] = 'generic_block'
            kwargs['name_scope'] = generic_name_scope
            self.generic_net = GenericNet("GenericNet", **kwargs)

    def call(self, inputs, train_phase):
        v, x, t = inputs

        if self.x_conv_net is not None:
            v = self.v_conv_net(v, train_phase)
            x = self.x_conv_net(x, train_phase)

        scale, translation, transformation = self.generic_net([v, x, t],
                                                              train_phase)

        return scale, translation, transformation
