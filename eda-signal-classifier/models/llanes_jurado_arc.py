# import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
import scipy.interpolate as sc_int 

### Deep Learning libraries
import tensorflow as tf
import keras

import keras.backend as K
from tensorflow.keras.optimizers import RMSProp
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Activation,
    Dropout,
    Dense,
    Flatten,
    LSTM,
    Input,
    GRU,
    BatchNormalization,
    Lambda,
    Add,
    concatenate,
    Reshape)
from keras.layers.merge import Concatenate
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau,TensorBoard
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.losses import binary_crossentropy

from tensorflow.keras.losses import CategoricalCrossentropy as cce_loss
from tensorflow.keras.metrics import CategoricalCrossentropy as cce_metric, CategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def model_recognition(window, dropout_value = 0.05):
    
    filter_size = 32 
    kernel_size = 5
    
    inputs = Input((window, 1))
    
    # ### LSTM layers ###
    
    lstm1 = LSTM(16, activation="tanh", return_sequences=True)(inputs)
    lstm1 = BatchNormalization()(lstm1)
    lstm1_drop = Dropout(dropout_value)(lstm1)

    lstm2 = LSTM(16, activation="tanh", return_sequences=True)(lstm1_drop)
    lstm2 = BatchNormalization()(lstm2)
    lstm2_drop = Dropout(dropout_value)(lstm2)

    # ### 1D CNN layers ###
    
    conv1 = Conv1D(filter_size, kernel_size, activation='relu', padding = 'same',
                   kernel_initializer = 'he_normal')(lstm2_drop)
    conv1 = BatchNormalization()(conv1)
    conv1_1 = Conv1D(filter_size, kernel_size, activation='relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv1)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_2 = Conv1D(filter_size, kernel_size, activation='relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    drop1 = Dropout(dropout_value)(conv1_2)
    add1 = Add()([drop1, conv1])
    max1 = MaxPooling1D(pool_size=2)(add1)
    
    conv2 = Conv1D(2*filter_size, kernel_size, activation='relu', padding = 'same',
                   kernel_initializer = 'he_normal')(max1)
    conv2 = BatchNormalization()(conv2)
    conv2_1 = Conv1D(2*filter_size, kernel_size, activation='relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv2)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_2 = Conv1D(2*filter_size, kernel_size, activation='relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    drop2 = Dropout(dropout_value)(conv2_2)
    add1 = Add()([drop2, conv2])
    max2 = MaxPooling1D(pool_size=2)(add1)
    
    conv3 = Conv1D(4*filter_size, kernel_size, activation='relu', padding = 'same',
                   kernel_initializer = 'he_normal')(max2)
    conv3 = BatchNormalization()(conv3)
    conv3_1 = Conv1D(4*filter_size, kernel_size, activation='relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv3)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_2 = Conv1D(4*filter_size, kernel_size, activation='relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    drop3 = Dropout(dropout_value)(conv3_2)
    add3 = Add()([drop3, conv3])
    max3 = MaxPooling1D(pool_size=2)(add3)

    conv4 = Conv1D(8*filter_size, kernel_size, activation='relu', padding = 'same',
                   kernel_initializer = 'he_normal')(max3)
    conv4 = BatchNormalization()(conv4)
    conv4_1 = Conv1D(8*filter_size, kernel_size, activation='relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv4)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_2 = Conv1D(8*filter_size, kernel_size, activation='relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv4_1)
    conv4_2 = BatchNormalization()(conv4_2)
    drop4 = Dropout(dropout_value)(conv4_2)
    add4 = Add()([drop4, conv4])
    max4 = MaxPooling1D(pool_size=2)(add4)

    conv5 = Conv1D(16*filter_size, kernel_size, activation='relu', padding = 'same',
                   kernel_initializer = 'he_normal')(max4)
    conv5 = BatchNormalization()(conv5)
    conv5_1 = Conv1D(16*filter_size, kernel_size, activation='relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv5)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_2 = Conv1D(16*filter_size, kernel_size, activation='relu', padding = 'same',
                   kernel_initializer = 'he_normal')(conv5_1)
    conv5_2 = BatchNormalization()(conv5_2)
    drop5 = Dropout( dropout_value )(conv5_2)
    add5 = Add()([drop5, conv5])
    max5 = MaxPooling1D(pool_size=2)(add5)
        
    flat = Flatten()(max5)
    
    # ### Dense layers ###
    
    dense1 = Dense(256, activation = "relu")(flat)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    
    dense2 = Dense(16, activation="relu")(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.5)(dense2)
    
    dense3 = Dense(1, activation="sigmoid")(dense2) 
    
    model = Model(inputs, dense3)


# @tf.keras.utils.register_keras_serializable()
# class GenPhiloTextA(tf.keras.Model):
#     def __init__(self, emb_dim=32, n_a=128, n_unique=26, dense_layers_dims=[26], lambda_=1, drop_prob=0.0, normalize=False, **kwargs):
#         super(GenPhiloTextA, self).__init__(**kwargs)
#         self.emb_dim = emb_dim
#         self.n_a = n_a
#         self.n_unique = n_unique
#         self.dense_layers_dims = dense_layers_dims
#         self.lambda_ = lambda_
#         self.drop_prob = drop_prob

#         # number of time steps or length of longest sequences/training example
#         self.n_dense_layers = len(dense_layers_dims)
#         self.normalize = normalize

#         # instantiate layers
#         self.character_lookup = Embedding(n_unique, emb_dim, name='character-lookup', embeddings_regularizer=L2(lambda_))
#         self.lstm_layer = LSTM(units=n_a, return_sequences=True, return_state=True, name='lstm-layer')
#         self.dense_layers = [Dense(units=dim, name=f'dense-layer-{i}', kernel_regularizer=L2(lambda_)) for i, dim, in enumerate(dense_layers_dims)]
#         self.norm_layers = [BatchNormalization(name=f'norm-layer-{i}') for i in range(len(dense_layers_dims) - 1)]
#         self.act_layers = [Activation(activation=tf.nn.relu, name=f'act-layer-{i}') for i in range(len(dense_layers_dims) - 1)]
#         self.drop_layers = [Dropout(drop_prob, name=f'drop-layer-{i}') for i in range(len(dense_layers_dims) - 1)]

#     def call(self, inputs, h=None, c=None, return_state=False, training=False):
#         """
#         args:
#             inputs - 
#             states - is a list containing initial state or current state
#             h and c
#             return_state -
#             training - 
#         """

#         # get batch of training examples
#         X = inputs
#         embeddings = self.character_lookup(X, training=training)

#         # check if states are empty. Note h and c will of
#         # course be None during training so this is the reason
#         # why self.lstm_layer.get_initial_state() is called
#         # but during inference state will be provided using loop
#         if h is None and c is None:
#             h, c = self.lstm_layer.get_initial_state(embeddings)
#         hs, h, c = self.lstm_layer(embeddings, initial_state=[h, c], training=training)

#         temp = hs
#         for i in range(self.n_dense_layers - 1):
#             temp = self.dense_layers[i](temp, training=training)
                
#             # if normalize is false do not permit passing temp 
#             # to batch normalization layer
#             if self.normalize == True:
#                 temp = self.norm_layers[i](temp, training=training)

#             # note model only passes the activation to dropout 
#             # during training
#             temp = self.act_layers[i](temp, training=training)
#             temp = self.drop_layers[i](temp, training=training)

#         # output logits will be a (m, Ty, n_unique) 
#         logits = self.dense_layers[-1](temp, training=training)

#         # only is the logits, h, and c are returned during 
#         # sampling but during training only logits are 
#         # returned per batch of inputs
#         return (logits, h, c) if return_state == True else logits
    
#     def get_config(self):
#         config = super(GenPhiloTextA, self).get_config()
#         config['emb_dim'] = self.emb_dim
#         config['n_a'] = self.n_a
#         config['n_unique'] = self.n_unique
#         config['dense_layers_dims'] = self.dense_layers_dims
#         config['lambda_'] = self.lambda_
#         config['drop_prob'] = self.drop_prob
#         config['normalize'] = self.normalize

#         return config


if __name__ == "__main__":
    # llanes-jurado et al. (2023) has stated that the optimizer they used was RMSProp
    # with a learning rate of 5e-5 or 0.00005, a batch size of 16, a metric that uses the
    # Dice-Sorensen coefficient, and with an early stopping threshold of 30 epochs 
    # meaning if the Dice-Sorensen coefficient metric value does not better in 30 epochs then
    # the model on the last epoch with the best Dice-Sorensen coefficient metric value will be
    # saved

    # Initialize the model
    # why they used a 5 second window is because each segment has labels [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    # where only the last 0.5s of the 5 second window/segment or epoch is labeled an artifact
    EDABE_LSTM_1DCNN_Model = model_recognition(window=5*128, dropout_value=0.05)

    # # hyperparameters
    # m = 20000
    # T_x = 100
    # n_unique = 57
    # n_a = 64
    # emb_dim = 32
    # dense_layers_dims = [n_unique]
    # lambda_ = 0.8
    # drop_prob = 0.4
    # learning_rate = 1e-3
    # epochs = 100
    # batch_size = 512
    # normalize = False

    # # note X becomes (m, T_x, n_features) when fed to embedding layer
    # X = np.random.randint(0, n_unique, size=(m, T_x))

    # # we have to match the output of the prediction of our 
    # # model which is a list of (100, 26) values. So instead of a 3D matrixc
    # # we create a list fo 2D matrices of shape (100, 26)
    # Y = [np.random.rand(m, n_unique) for _ in range(T_x)]

    # # one hot encode our dummy (T_y, m, n_unique) probabilities
    # Y = [tf.one_hot(tf.argmax(y, axis=1), depth=n_unique) for y in Y]
    
    # # test for computing loss with (m, T_y, n_unique) predictions
    # Y_true = tf.reshape(Y, shape=(m, T_x, n_unique))
    # dummy_logits = np.random.randn(m, T_x, n_unique)
    # loss = cce_loss(from_logits=True)(dummy_logits, Y_true)
    # print(f"computed test loss: {loss}")

    # # initialize hidden and cell states to shape (m, n_units)
    # h_0 = np.zeros(shape=(m, n_a))
    # c_0 = np.zeros(shape=(m, n_a))

    # # instantiate custom model
    # model = GenPhiloTextA(emb_dim=emb_dim, n_a=n_a, n_unique=n_unique, dense_layers_dims=dense_layers_dims, lambda_=lambda_, drop_prob=drop_prob, normalize=normalize)
    # # model = GenPhiloTextB(emb_dim=emb_dim, n_a=n_a, n_unique=n_unique, T_x=T_x, dense_layers_dims=dense_layers_dims, lambda_=lambda_, drop_prob=drop_prob, normalize=normalize)

    # # define loss, optimizer, and metrics then compile
    # opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
    # loss = cce_loss(from_logits=True)
    # metrics = [CategoricalAccuracy(), cce_metric(from_logits=True)]    
    # model.compile(optimizer=opt, loss=loss, metrics=metrics)
    # model(X)
    # # model([X, h_0, c_0])
    # model.summary()

    # # define checkpoint and early stopping callback to save
    # # best weights at each epoch and to stop if there is no improvement
    # # of validation loss for 10 consecutive epochs
    # weights_path = f"../saved/weights/test_{model.name}" + "_{epoch:02d}_{val_loss:.4f}.h5"
    # checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
    # stopper = EarlyStopping(monitor='val_loss', patience=10)
    # callbacks = [checkpoint, stopper]

    # # begin training test model
    # history = model.fit(X, Y_true, 
    #     epochs=epochs,
    #     batch_size=batch_size, 
    #     callbacks=callbacks,
    #     validation_split=0.3,
    #     verbose=2,)
    
    # # history = model.fit([X, h_0, c_0], Y_true, 
    # #     epochs=epochs,
    # #     batch_size=batch_size, 
    # #     callbacks=callbacks,
    # #     validation_split=0.3,
    # #     verbose=2,)
    
    # # save model
    # # model.save_weights('../saved/weights/test_model_gen_philo_text.h5', save_format='h5')
    # # model.save('../saved/models/test_model_b.h5', save_format='h5')