# -*- coding: utf-8 -*-
# @Time : 2021/8/4 17:08
# @Author : Raymond
# @File : se_text_cnn.py
# @Software: PyCharm

import squeeze_excitation_layer as se
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, AveragePooling1D, Dense, Concatenate, GlobalMaxPooling1D, LSTM, \
    Bidirectional, BatchNormalization, Dropout
from tensorflow.keras import Model


class KmerSETextCNN(Model):
    """SE-BLTCNN model

        Define the structure of the SE-BLTCNN model.

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
            bilstm_layers: List. The bilstm layers that comprise the BLSTM
            component in the model.
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
                 lstm_units,
                 conv1d_filters,
                 kernel_sizes=[1, 2, 3],
                 kernel_regularizer=None,
                 last_activation='softmax',
                 embedded_input=False,
                 pre_avg_window=1
                 ):
        """Initialize function of SE-BLTCNN model

                Initialize the SE-BLTCNN object.

                Params:
                   class_num: The number of classes in the classification problem.
                   maxlen: The maximum input length of the input sequence.
                   input_dim: Integer. Size of the vocabulary. When embedded_input=True,
                   it would be ignored.
                   embedding_dims: Integer. Dimension of the dense embedding.
                   lstm_units: Integer. The number of neurons in the hidden layer of the
                   LSTM block.
                   conv1d_filters: Integer. The number of filters in each filter size.
                   kernel_sizes: List, eg: [1,2,3]. The window size of 3 conv1d layers
                   in the TextCNN component.
                   kernel_regularizer: eg: tf.keras.regularizers.l2(0.001)
                   last_activation: The activation function used in the output layer.
                   embedded_input: Truth value. True, if the input has already been
                   embedded into a feature space default：False.
                   pre_avg_window: Integer. The window size of the very first average
                   pooling layer.
                Returns:
                    None
           """

        super(KmerSETextCNN, self).__init__()

        self.class_num = class_num
        self.maxlen = maxlen
        self.embedding_dims = embedding_dims
        self.kernel_sizes = kernel_sizes
        self.embedded_input = embedded_input
        self.kmer = pre_avg_window

        if not self.embedded_input:
            self.embedding = Embedding(input_dim=input_dim, output_dim=embedding_dims, input_length=maxlen)

        # ============ 模型前半部分——BiLSTM部分 的定义 ==============
        self.bilstm_layers = []
        self.bilstm_layers.append((Bidirectional(LSTM(units=lstm_units, return_sequences=True)), BatchNormalization(),
                                   Dropout(0.5)))
        self.bilstm_layers.append((Bidirectional(LSTM(units=lstm_units, return_sequences=True)), BatchNormalization(),
                                   Dropout(0.5)))

        # ============ 模型后半部分——text CNN部分的定义 =============
        # 在将embedding层输出送入TextCNN之前，增加一个AveragePooling1D层，该层按照kmer长度的窗口做一维平均池化。
        if pre_avg_window > 1:
            self.prior_avgpool = AveragePooling1D(pool_size=pre_avg_window, strides=pre_avg_window // 3 * 2)

        self.conv1s = []
        self.maxpools = []
        for kernel_size in kernel_sizes:
            # original filters=128
            self.conv1s.append(
                (Conv1D(filters=conv1d_filters, kernel_size=kernel_size, activation='relu',
                        kernel_regularizer=kernel_regularizer),
                 se.Squeeze_excitation_layer(conv1d_filters, conv1d_filters//2)))

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

        # 输入到两层BiLSTM
        i = 0
        for bilstm, bn, dropout in self.bilstm_layers:
            if i == 0:
                h = bilstm(emb)
            else:
                h = bilstm(h)
            h = bn(h)
            h = dropout(h)
            i += 1

        # 在将embedding层输出送入TextCNN之前，增加一个AveragePooling1D层，该层按照kmer长度的窗口做一维平均池化。
        if self.kmer > 1:
            h = self.prior_avgpool(h)

        # 输入到text CNN
        conv1s = []
        for i in range(len(self.kernel_sizes)):
            c = self.conv1s[i][0](h)  # (batch_size, maxlen-kernel_size+1, filters)
            c = self.conv1s[i][1](c)  # Squeeze & Excitation layer
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
        # 自定义函数，在调用model.summary()之前调用
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")
        _ = self.call(inputs)
