# -*- coding: utf-8 -*-
# @Time : 2021/8/4 17:08
# @Author : Raymond
# @File : textcnn.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, AveragePooling1D, Dense, Concatenate, GlobalMaxPooling1D
from tensorflow.keras import Model


class KmerTextCNN(Model):
    """Textcnn-based model

    Define the structure of the textcnn-based model.

    Attributes:
        class_num: The number of classes in the classification problem.
        maxlen: The maximum input length of the input sequence.
        embedding_dims: Integer. Dimension of the dense embedding.
        kernel_sizes: list, eg: [1,2,3]. The window size of 3 conv1d layers
        in the TextCNN component.
        embedded_input: Truth value. True, if the input has already been
        embedded into a feature space default：False.
        kmer: Integer. The window size of the very first average
        pooling layer.
        prior_avgpool: AveragePooling1D。The 1D maxpooling layer of
        the downsampling component in the model.
        conv1s: List. The conv1d layers that comprise the TextCNN component
        in the model.
        maxpools: List. The maxpooling layers in the TextCNN component.
    """

    def __init__(self,
                 class_num,
                 maxlen,
                 input_dim,
                 embedding_dims,
                 conv1d_filters,
                 kernel_sizes=[1, 2, 3],
                 kernel_regularizer=None,
                 last_activation='softmax',
                 embedded_input=False,
                 pre_avg_window=1
                 ):
        super(KmerTextCNN, self).__init__()
        self.maxlen = maxlen
        self.embedding_dims = embedding_dims
        self.kernel_sizes = kernel_sizes
        self.class_num = class_num
        self.embedded_input = embedded_input
        self.kmer = pre_avg_window
        if not self.embedded_input:
            self.embedding = Embedding(input_dim=input_dim, output_dim=embedding_dims, input_length=maxlen)

        # ============ text CNN部分的定义 =============
        # 在将embedding层输出送入TextCNN之前，增加一个AveragePooling1D层，该层按照kmer长度的窗口做一维平均池化。
        if pre_avg_window > 1:
            self.prior_avgpool = AveragePooling1D(pool_size=pre_avg_window, strides=pre_avg_window // 3 * 2)

        self.conv1s = []
        self.maxpools = []
        for kernel_size in kernel_sizes:
            # original filters=128
            self.conv1s.append(
                Conv1D(filters=conv1d_filters, kernel_size=kernel_size, activation='relu', kernel_regularizer=kernel_regularizer))
            self.maxpools.append(GlobalMaxPooling1D())
        self.classifier = Dense(class_num, activation=last_activation)

    def call(self, inputs, training=None, mask=None):
        if not self.embedded_input and len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextCNN must be 2, but now is %d' % len(inputs.get_shape()))
        elif self.embedded_input and len(inputs.get_shape()) != 3:
            raise ValueError('The rank of inputs of TextCNN must be 3, but now is %d' % len(inputs.get_shape()))

        if not self.embedded_input and inputs.get_shape()[1] != self.maxlen:
            raise ValueError(
                'The maxlen of inputs of TextCNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))

        if not self.embedded_input:
            emb = self.embedding(inputs)
        else:
            emb = inputs

        # 在将embedding层输出送入TextCNN之前，增加一个AveragePooling1D层，该层按照kmer长度的窗口做一维平均池化。
        if self.kmer > 1:
            h = self.prior_avgpool(emb)

        # 输入到text CNN
        conv1s = []
        for i in range(len(self.kernel_sizes)):
            c = self.conv1s[i](h)  # (batch_size, maxlen-kernel_size+1, filters)
            c = self.maxpools[i](c)  # # (batch_size, filters)
            conv1s.append(c)
        x = Concatenate()(conv1s)  # (batch_size, len(self.kernel_sizes)*filters)
        output = self.classifier(x)
        return output

    def build_graph(self, input_shape):
        """Call before running model.summary()

        Call this function before running model.summary(), so that model.summary()
        can work when model.fit() has not been run.
        """
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")
        _ = self.call(inputs)
