"""
tensorflow/network.py

Tensorflow implementation of the network used to train the L2HMC sampler.
"""
from __future__ import absolute_import, print_function, division, annotations

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import (
    Layer, Add, Dropout, Dense, Flatten, Conv2D, MaxPooling2D,
    BatchNormalization, Input
)

ACTIVATION_FNS = {
}

