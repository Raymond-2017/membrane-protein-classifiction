# -*- coding: utf-8 -*-
# @Time : 2021/8/4 17:08
# @Author : Raymond
# @File : main.py
# @Software: PyCharm

# ===================== set random  ===========================
import os
import time

import numpy as np
import tensorflow as tf
import random as rn

def setup_seed(seed):
    rn.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first

setup_seed(0)
# =============================================================
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 占用GPU80%的显存
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from se_text_cnn import KmerSETextCNN

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def checkout_dir(dir_path, do_delete=False):
    import shutil
    if do_delete and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        print(dir_path, 'make dir ok')
        os.makedirs(dir_path)


'''
    构建模型helper。帮助构建模型，定义和管理各种回调函数。    
    :param maxlen: 文本最大长度
    :param max_features: 词典大小
    :param embedding_dims: embedding维度大小
    :param kernel_sizes: 滑动卷积窗口大小的list, eg: [1,2,3]
    :param kernel_regularizer: eg: tf.keras.regularizers.l2(0.001)
    :param class_num: 类别个数。8种膜蛋白。 注意：一定要填对，不然会训练中会出现loss为NaN的情况!
    :param embedded_input: 是否是已经嵌入过的输入数据。 default：False
'''


class ModelHepler:
    def __init__(self, class_num, maxlen, max_features, embedding_dims, epochs, batch_size, embedded_input,
                 pre_avg_window, lstm_units, conv1d_filters):
        self.class_num = class_num
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.pre_avg_window = pre_avg_window
        self.lstm_units = lstm_units
        self.conv1d_filters = conv1d_filters
        self.epochs = epochs
        self.batch_size = batch_size
        self.callback_list = []
        self.embedded_input = embedded_input
        print('Bulid Model...')
        self.create_model()

    def create_model(self):
        model = KmerSETextCNN(maxlen=self.maxlen,
                              max_features=self.max_features,
                              embedding_dims=self.embedding_dims,
                              class_num=self.class_num,
                              lstm_units=self.lstm_units,
                              conv1d_filters=self.conv1d_filters,
                              kernel_sizes=kernel_sizes,
                              kernel_regularizer=None,
                              last_activation='softmax',
                              embedded_input=self.embedded_input,
                              pre_avg_window=self.pre_avg_window)
        model.compile(
            # optimizer='adam',
            optimizer=tf.keras.optimizers.Adam(),
            # optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
        )

        if not self.embedded_input:
            input_shape = (None, self.maxlen)  # (batch_size, maxlen) when input has not been embedded yet.
        else:
            input_shape = (None, self.maxlen, self.embedding_dims)  # (batch_size, maxlen, filters)
            # when input has been embedded
        model.build_graph(input_shape)
        model.summary()
        self.model = model

    def get_callback(self, use_early_stop=True, tensorboard_log_dir='logs\\TextCNN-epoch-5',
                     checkpoint_path="save_model_dir\\cp-model.ckpt", lr_decay=None):
        callback_list = []
        if use_early_stop:
            # EarlyStopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=6, mode='min')
            callback_list.append(early_stopping)
        if checkpoint_path is not None:
            # save model
            checkpoint_dir = os.path.dirname(checkpoint_path)
            checkout_dir(checkpoint_dir, do_delete=True)
            # 创建一个保存模型权重的回调
            cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          monitor='val_loss',
                                          mode='min',
                                          save_best_only=True,
                                          save_weights_only=True,
                                          verbose=1,
                                          period=1,
                                          )
            callback_list.append(cp_callback)
        if lr_decay is not None:
            callback_list.append(lr_decay)
        if tensorboard_log_dir is not None:
            # tensorboard --logdir logs\\TextCNN-epoch-5
            checkout_dir(tensorboard_log_dir, do_delete=True)
            tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
            callback_list.append(tensorboard_callback)
        self.callback_list = callback_list

    def fit(self, x_train, y_train, x_val, y_val):
        print('Train...')
        self.model.fit(x_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=1,  # 0=silent, 1=progress bar, 2=one line per epoch (log file in production environment)
                       callbacks=self.callback_list,
                       validation_data=(x_val, y_val))  # Data on which to evaluate the loss and any model metrics
        # at the end of each epoch.

    def load_model(self, checkpoint_path):
        checkpoint_dir = os.path.dirname((checkpoint_path))
        latest = tf.train.latest_checkpoint(checkpoint_dir)  # 加载最新保存的模型
        print('restore model name is : ', latest)
        # 创建一个新的模型实例
        # model = self.create_model()
        # 加载以前保存的权重
        self.model.load_weights(latest)


# ================  params =========================
class_num = 8
maxlen = 1024
embedding_dims = 20  # embedded_input为True时，不发挥作用。
epochs = 30
max_features = 5000  # embedded_input为True时，不发挥作用。
pre_avg_window = 12
embedded_input = True
kernel_sizes = [3, 4, 5]

MODEL_NAME = 'TextCNN-epoch-50-emb-20'

use_early_stop = False
tensorboard_log_dir = 'logs\\{}'.format(MODEL_NAME)

# checkpoint_path = "save_model_dir\\{}\\cp-{epoch:04d}.ckpt".format(MODEL_NAME, '')
checkpoint_dir = 'save_model_dir\\' + MODEL_NAME + '-comb{:02d}\\'
best_ckpt_path = ''

# ==================== model selection param =========================
batch_sizes = [32]  # number of training examples are fed to one iteration
LSTM_units = [256]  # number of units in LSTM
conv1d_filters = [64]  # number of filters in text-CNN layer
lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='min')
#  ====================================================================


print('Loading data...')
# 加载膜蛋白分类训练数据
# We load our dataset from files, the procedure of saving file was done by dataset_utils.py
x_train = np.load('..\\data\\X.npy')
x_train = x_train[:, :maxlen, :]
y_train = np.load('..\\data\\Y.npy')

# Split the dataset to training set and validation set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=33, stratify=y_train)
# %%
print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)
print('y_train shape:', y_train.shape)
print('y_val shape:', y_val.shape)
# %%

x_test = np.load('..\\data\\X_test.npy')
x_test = x_test[:, :maxlen, :]
# read test set from files
y_test = np.load('..\\data\\Y_test.npy')


def model_selection():
    """
    We search for optimal hyperparameters for model using grid search.
    """
    i = 0
    for batch_size in batch_sizes:
        for units in LSTM_units:
            for filter in conv1d_filters:
                i += 1
                # ++++++++++++++++++++++++++++++++++++
                # 保存训练模型使用的参数
                global checkpoint_dir
                checkpoint_dir = checkpoint_dir.format(i)

                ckpt_path = checkpoint_dir + 'cp-{epoch:02d}.ckpt'
                checkout_dir(os.path.dirname(ckpt_path), do_delete=True)

                param_dict = {'maxlen': maxlen, 'epochs': epochs, 'use_early_stop': use_early_stop,
                              'MODEL_NAME': MODEL_NAME,
                              'tensorboard_log_dir': tensorboard_log_dir, 'checkpoint_dir': checkpoint_dir,
                              'kernel_sizes': kernel_sizes,
                              'batch_sizes': batch_sizes, 'LSTM_units': LSTM_units, 'conv1d_filters': conv1d_filters}
                uuid_str = time.strftime("%Y-%m-%d-%H_%M", time.localtime())
                tmp_file_name = 'hyperparam-%s.txt' % uuid_str
                with open(checkpoint_dir + tmp_file_name, 'w+') as f:
                    f.write("{}\n\n".format(str(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()))))
                    for k, v in param_dict.items():
                        f.write(f'{k}: {v}\n')
                # ++++++++++++++++++++++++++++++++++++
                model_hepler = ModelHepler(class_num=class_num,
                                           maxlen=maxlen,
                                           max_features=max_features,
                                           embedding_dims=embedding_dims,
                                           epochs=epochs,
                                           batch_size=batch_size,
                                           embedded_input=embedded_input,
                                           pre_avg_window=pre_avg_window,
                                           lstm_units=units,
                                           conv1d_filters=filter)

                model_hepler.get_callback(use_early_stop=use_early_stop, tensorboard_log_dir=tensorboard_log_dir,
                                          checkpoint_path=ckpt_path, lr_decay=lr_decay)
                model_hepler.fit(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
                # 保存最后一轮的模型权重
                model_hepler.model.save_weights(checkpoint_dir + 'model_final_weights.h5')
                print('Test...')
                test_score = model_hepler.model.evaluate(x_test, y_test,
                                                         batch_size=batch_size)
                print("test loss:", test_score[0], "test accuracy", test_score[1])
                # %%
                pred = model_hepler.model.predict(x_test, batch_size=256)
                pred = np.argmax(pred, axis=1)
                print(pred.shape)

                report = classification_report(y_true=y_test, y_pred=pred, digits=4)
                print(report)
                with open('classification_report%d.txt' % i, 'w') as f:
                    f.write("test loss: {:f}, test accuracy: {:f}\n".format(test_score[0], test_score[1]))
                    f.write(report)
                # %%

                # === 混淆矩阵：真实值与预测值的对比 ===
                # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
                con_mat = confusion_matrix(y_test, pred)
                np.save('image\\con_mat.npy', con_mat)

                con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
                con_mat_norm = np.around(con_mat_norm, decimals=2)
                np.save('image\\con_mat_norm.npy', con_mat_norm)

                # === plot confusion matrix ===
                plt.figure(figsize=(8, 8))
                protein_types = ['Single-pass\ntype I  ', 'Single-pass\ntype II  ', 'Single-pass\ntype III  ',
                                 'Single-pass\ntype IV  ', 'Multipass', 'Lipid-chain-anchor', 'GPI-anchor',
                                 'Peripheral']
                sns.heatmap(con_mat_norm, annot=True, cmap='Blues', xticklabels=protein_types,
                            yticklabels=protein_types, fmt='g', vmax=1)

                plt.ylim(0, 8)
                plt.xticks(rotation=35, ha='right')
                plt.xlabel('Predicted labels')
                plt.ylabel('True labels')
                plt.title('(c) SE embedded BiLSTM-TextCNN')

                plt.subplots_adjust(left=0.225, bottom=0.21, right=0.96, top=0.94)
                plt.savefig('image\\sns_heatmap_cmap.jpg')
                plt.show()


model_selection()

i = 0
for batch_size in batch_sizes:
    for units in LSTM_units:
        for filter in conv1d_filters:
            i += 1
            # 重新评估模型
            model_hepler = ModelHepler(class_num=class_num,
                                       maxlen=maxlen,
                                       max_features=max_features,
                                       embedding_dims=embedding_dims,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       embedded_input=True,
                                       pre_avg_window=pre_avg_window,
                                       lstm_units=units,
                                       conv1d_filters=filter)

            ckpt_dir = checkpoint_dir.format(i)
            model_hepler.load_model(checkpoint_path=ckpt_dir)

            loss, acc = model_hepler.model.evaluate(x_test, y_test, verbose=2)
            print("Restored model from {}. loss: {:.4f}, accuracy: {:5.2f}%\n".format(ckpt_dir, loss, 100 * acc))
            # %%
            pred = model_hepler.model.predict(x_test, batch_size=256)
            pred = np.argmax(pred, axis=1)
            print(pred.shape)
            report = classification_report(y_true=y_test, y_pred=pred, digits=4)
            print(report)
            with open('classification_report%d.txt' % i, 'a') as f:
                f.write(
                    "\nRestored model from {}. loss: {:.4f}, accuracy: {:5.2f}%\n".format(ckpt_dir, loss, 100 * acc))
                f.write(report)
            # %%

            # 保存模型为PB文件。
            tf.saved_model.save(model_hepler.model, ckpt_dir)

