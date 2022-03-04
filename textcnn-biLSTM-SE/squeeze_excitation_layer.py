# -*- coding: utf-8 -*-
# @Time : 2021/8/15 10:32
# @Author : Raymond
# @File : squeeze_excitation_layer.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np


class Squeeze_excitation_layer(tf.keras.Model):
    """ Squeeze-and-Excitation block

        Define the Squeeze and Excitation block, the channel attention mechanism
        used in the SE-BLTCNN model.
    """
    def __init__(self, filters, filters_sq):
        """Initialize function

            Initialize the SE layer object.

            Params:
                filters: Integer. The number of channels before dimensionality
                reduction operation.
                filters_sq: Integer. The number of channels after
                dimensionality reduction operation.
            Returns:
                None
        """
        # filters 降维前的通道数。  filters_sq 降维后的通道数。
        super().__init__()
        self.filter_sq = filters_sq
        self.avgpool = tf.keras.layers.GlobalAveragePooling1D()
        self.dense1 = tf.keras.layers.Dense(filters_sq)
        self.leaky_relu = tf.keras.layers.Activation(tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(filters)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, inputs):
        squeeze = self.avgpool(inputs)

        excitation = self.dense1(squeeze)
        excitation = self.leaky_relu(excitation)
        excitation = self.dense2(excitation)
        excitation = self.sigmoid(excitation)
        excitation = tf.keras.layers.Reshape((1, inputs.shape[-1]))(excitation)

        scale = inputs * excitation

        return scale

if __name__ == '__main__':
    SE = Squeeze_excitation_layer(32, 16)
    inputs = np.ones((1, 32, 32), dtype=np.float32)
    print(SE(inputs).shape)
