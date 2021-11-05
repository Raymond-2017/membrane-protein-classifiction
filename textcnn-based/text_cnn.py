# -*- coding: utf-8 -*-
# @Time : 2020/4/20 14:44
# @Author : zdqzyx
# @File : textcnn.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, AveragePooling1D, Dense, Concatenate, GlobalMaxPooling1D
from tensorflow.keras import Model


class KmerTextCNN(Model):
    '''
   :param maxlen: 文本最大长度
   :param max_features: 词典大小
   :param embedding_dims: embedding维度大小
   :param kernel_sizes: 滑动卷积窗口大小的list, eg: [1,2,3]
   :param kernel_regularizer: eg: tf.keras.regularizers.l2(0.001)
   :param class_num: 类别个数。8种膜蛋白。 注意：一定要填对，不然会训练中会出现loss为NaN的情况!
   :param last_activation: 最后一层的激活函数
   :param pre_avg_window: 预先（第一层）所作平均池化的窗口长度
    '''

    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
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
            self.embedding = Embedding(input_dim=max_features, output_dim=embedding_dims, input_length=maxlen)

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

    '''
    如果需要使用到其他Layer结构或者Sequential结构，需要在__init__()函数里赋值
    在model没有fit前，想调用summary函数时显示模型各层shape时，则需要自定义一个函数去build下模型，类似下面代码中的build_graph函数
    summary()显示shape顺序，是按照__init__()里layer赋值的顺序
    ————————————————
    版权声明：本文为CSDN博主「布鲁克泰勒」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
    原文链接：https://blog.csdn.net/sinat_18127633/article/details/105860790
    '''

    def build_graph(self, input_shape):
        '''自定义函数，在调用model.summary()之前调用
        '''
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")
        _ = self.call(inputs)