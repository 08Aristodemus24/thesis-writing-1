import tensorflow as tf
from tensorflow import GradientTape
from tensorflow.keras import Sequential, Model, Input
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
    Concatenate,
    Reshape,
    Conv1D,
    MaxPooling1D)
from tensorflow.keras.metrics import BinaryCrossentropy as bce_metric, BinaryAccuracy, Precision, Recall, AUC, F1Score

# @tf.keras.utils.register_keras_serializable()
# class LSTM_CNN(tf.keras.Model):
#     def __init__(self, window_size=5 * 128, n_a=16, drop_prob=0.05, filter_size=32, kernel_size=5, **kwargs):
#         super(LSTM_CNN, self).__init__(**kwargs)
#         self.window_size = window_size
#         self.n_a = n_a
#         self.drop_prob = drop_prob
#         self.filter_size = filter_size
#         self.kernel_size = kernel_size

#         # LSTM layers
#         self.lstm_layer_1 = LSTM(n_a, activation=tf.nn.tanh, return_sequences=True)
#         self.lstm_norm_1 = BatchNormalization()
#         self.lstm_drop_1 = Dropout(drop_prob)

#         self.lstm_layer_2 = LSTM(n_a, activation=tf.nn.tanh, return_sequences=True)
#         self.lstm_norm_2 = BatchNormalization()
#         self.lstm_drop_2 = Dropout(drop_prob)

#         # 1D CNN layers
#         self.conv_lvl_1_layer_1 = Conv1D(filter_size, kernel_size, activation=tf.nn.relu, padding='same', kernel_initializer='he_normal')
#         self.conv_lvl_1_norm_1 = BatchNormalization()
#         self.conv_lvl_1_layer_2 = Conv1D(filter_size, kernel_size, activation=tf.nn.relu, padding='same', kernel_initializer='he_normal')
#         self.conv_lvl_1_norm_2 = BatchNormalization()
#         self.conv_lvl_1_layer_3 = Conv1D(filter_size, kernel_size, activation=tf.nn.relu, padding='same', kernel_initializer='he_normal')
#         self.conv_lvl_1_norm_3 = BatchNormalization()
#         self.conv_lvl_1_drop_1 = Dropout(drop_prob)
#         self.add_lvl_1 = Add()
#         self.max_lvl_1 = MaxPooling1D(pool_size=2)
        
#         self.conv_lvl_2_layer_1 = Conv1D(2 * filter_size, kernel_size, activation=tf.nn.relu, padding='same', kernel_initializer='he_normal')
#         self.conv_lvl_2_norm_1 = BatchNormalization()
#         self.conv_lvl_2_layer_2 = Conv1D(2 * filter_size, kernel_size, activation=tf.nn.relu, padding='same', kernel_initializer='he_normal')
#         self.conv_lvl_2_norm_2 = BatchNormalization()
#         self.conv_lvl_2_layer_3 = Conv1D(2 * filter_size, kernel_size, activation=tf.nn.relu, padding='same', kernel_initializer='he_normal')
#         self.conv_lvl_2_norm_3 = BatchNormalization()
#         self.conv_lvl_2_drop_1 = Dropout(drop_prob)
#         self.add_lvl_2 = Add()
#         self.max_lvl_2 = MaxPooling1D(pool_size=2)
        
#         self.conv_lvl_3_layer_1 = Conv1D(4 * filter_size, kernel_size, activation=tf.nn.relu, padding='same', kernel_initializer='he_normal')
#         self.conv_lvl_3_norm_1 = BatchNormalization()
#         self.conv_lvl_3_layer_2 = Conv1D(4 * filter_size, kernel_size, activation=tf.nn.relu, padding='same', kernel_initializer='he_normal')
#         self.conv_lvl_3_norm_2 = BatchNormalization()
#         self.conv_lvl_3_layer_3 = Conv1D(4 * filter_size, kernel_size, activation=tf.nn.relu, padding='same', kernel_initializer='he_normal')
#         self.conv_lvl_3_norm_3 = BatchNormalization()
#         self.conv_lvl_3_drop_1 = Dropout(drop_prob)
#         self.add_lvl_3 = Add()
#         self.max_lvl_3 = MaxPooling1D(pool_size=2)

#         self.conv_lvl_4_layer_1 = Conv1D(8 * filter_size, kernel_size, activation=tf.nn.relu, padding='same', kernel_initializer='he_normal')
#         self.conv_lvl_4_norm_1 = BatchNormalization()
#         self.conv_lvl_4_layer_2 = Conv1D(8 * filter_size, kernel_size, activation=tf.nn.relu, padding='same', kernel_initializer='he_normal')
#         self.conv_lvl_4_norm_2 = BatchNormalization()
#         self.conv_lvl_4_layer_3 = Conv1D(8 * filter_size, kernel_size, activation=tf.nn.relu, padding='same', kernel_initializer='he_normal')
#         self.conv_lvl_4_norm_3 = BatchNormalization()
#         self.conv_lvl_4_drop_1 = Dropout(drop_prob)
#         self.add_lvl_4 = Add()
#         self.max_lvl_4 = MaxPooling1D(pool_size=2)

#         self.conv_lvl_5_layer_1 = Conv1D(16 * filter_size, kernel_size, activation=tf.nn.relu, padding='same', kernel_initializer='he_normal')
#         self.conv_lvl_5_norm_1 = BatchNormalization()
#         self.conv_lvl_5_layer_2 = Conv1D(16 * filter_size, kernel_size, activation=tf.nn.relu, padding='same', kernel_initializer='he_normal')
#         self.conv_lvl_5_norm_2 = BatchNormalization()
#         self.conv_lvl_5_layer_3 = Conv1D(16 * filter_size, kernel_size, activation=tf.nn.relu, padding='same', kernel_initializer='he_normal')
#         self.conv_lvl_5_norm_3 = BatchNormalization()
#         self.conv_lvl_5_drop_1 = Dropout(drop_prob)
#         self.add_lvl_5 = Add()
#         self.max_lvl_5 = MaxPooling1D(pool_size=2)
            
#         # flatten layer
#         self.flat_layer = Flatten()
        
#         # dense layers
#         self.dense_layer_1 = Dense(units=256, activation=tf.nn.relu)
#         self.dense_norm_1 = BatchNormalization()
#         self.dense_drop_1 = Dropout(0.5)
        
#         self.dense_layer_2 = Dense(units=16, activation=tf.nn.relu)
#         self.dense_norm_2 = BatchNormalization()
#         self.dense_drop_2 = Dropout(0.5)
        
#         self.dense_layer_3 = Dense(units=1, activation=tf.nn.sigmoid)

#         # extra metrics
#         self.prec_metric = Precision(name="precision")
#         self.rec_metric = Recall(name="recall")

#     def call(self, inputs, training=False):
#         # input shape will be a (m, window_size, n_f) (m, Tx, nf) which
#         # based on our data we know we can reshape into since we have will
#         # have a window size of 5 * 128 according to Llanes-Jurado et al. (2023)
#         # so m, 5 * 128, 1 where 1 will only be our number of features since 
#         # there will be no feature engineering required

#         # LSTM layers
#         lstm_out_1 = self.lstm_layer_1(inputs, training=training)
#         lstm_normed_1 = self.lstm_norm_1(lstm_out_1, training=training)
#         lstm_dropped_1 = self.lstm_drop_1(lstm_normed_1, training=training)

#         lstm_out_2 = self.lstm_layer_2(lstm_dropped_1, training=training)
#         lstm_normed_2 = self.lstm_norm_2(lstm_out_2, training=training)
#         lstm_dropped_2 = self.lstm_drop_2(lstm_normed_2, training=training)

#         # 1d CNN layers

#         # level 1
#         conv_lvl_1_out_1 = self.conv_lvl_1_layer_1(lstm_dropped_2, training=training)
#         conv_lvl_1_normed_1 = self.conv_lvl_1_norm_1(conv_lvl_1_out_1, training=training)
#         conv_lvl_1_out_2 = self.conv_lvl_1_layer_2(conv_lvl_1_normed_1, training=training)
#         conv_lvl_1_normed_2 = self.conv_lvl_1_norm_2(conv_lvl_1_out_2, training=training)
#         conv_lvl_1_out_3 = self.conv_lvl_1_layer_3(conv_lvl_1_normed_2, training=training)
#         conv_lvl_1_normed_3 = self.conv_lvl_1_norm_3(conv_lvl_1_out_3, training=training)
#         conv_lvl_1_dropped_1 = self.conv_lvl_1_drop_1(conv_lvl_1_normed_3, training=training)
#         conv_lvl_1_added_1 = self.add_lvl_1([conv_lvl_1_dropped_1, conv_lvl_1_normed_1], training=training)
#         conv_lvl_1_max_1 = self.max_lvl_1(conv_lvl_1_added_1, training=training)

#         # level 2
#         conv_lvl_2_out_1 = self.conv_lvl_2_layer_1(conv_lvl_1_max_1, training=training)
#         conv_lvl_2_normed_1 = self.conv_lvl_2_norm_1(conv_lvl_2_out_1, training=training)
#         conv_lvl_2_out_2 = self.conv_lvl_2_layer_2(conv_lvl_2_normed_1, training=training)
#         conv_lvl_2_normed_2 = self.conv_lvl_2_norm_2(conv_lvl_2_out_2, training=training)
#         conv_lvl_2_out_3 = self.conv_lvl_2_layer_3(conv_lvl_2_normed_2, training=training)
#         conv_lvl_2_normed_3 = self.conv_lvl_2_norm_3(conv_lvl_2_out_3, training=training)
#         conv_lvl_2_dropped_1 = self.conv_lvl_2_drop_1(conv_lvl_2_normed_3, training=training)
#         conv_lvl_2_added_1 = self.add_lvl_2([conv_lvl_2_dropped_1, conv_lvl_2_normed_1], training=training)
#         conv_lvl_2_max_1 = self.max_lvl_2(conv_lvl_2_added_1, training=training)

#         # level 3
#         conv_lvl_3_out_1 = self.conv_lvl_3_layer_1(conv_lvl_2_max_1, training=training)
#         conv_lvl_3_normed_1 = self.conv_lvl_3_norm_1(conv_lvl_3_out_1, training=training)
#         conv_lvl_3_out_2 = self.conv_lvl_3_layer_2(conv_lvl_3_normed_1, training=training)
#         conv_lvl_3_normed_2 = self.conv_lvl_3_norm_2(conv_lvl_3_out_2, training=training)
#         conv_lvl_3_out_3 = self.conv_lvl_3_layer_3(conv_lvl_3_normed_2, training=training)
#         conv_lvl_3_normed_3 = self.conv_lvl_3_norm_3(conv_lvl_3_out_3, training=training)
#         conv_lvl_3_dropped_1 = self.conv_lvl_3_drop_1(conv_lvl_3_normed_3, training=training)
#         conv_lvl_3_added_1 = self.add_lvl_3([conv_lvl_3_dropped_1, conv_lvl_3_normed_1], training=training)
#         conv_lvl_3_max_1 = self.max_lvl_3(conv_lvl_3_added_1, training=training)

#         # level 4
#         conv_lvl_4_out_1 = self.conv_lvl_4_layer_1(conv_lvl_3_max_1, training=training)
#         conv_lvl_4_normed_1 = self.conv_lvl_4_norm_1(conv_lvl_4_out_1, training=training)
#         conv_lvl_4_out_2 = self.conv_lvl_4_layer_2(conv_lvl_4_normed_1, training=training)
#         conv_lvl_4_normed_2 = self.conv_lvl_4_norm_2(conv_lvl_4_out_2, training=training)
#         conv_lvl_4_out_3 = self.conv_lvl_4_layer_3(conv_lvl_4_normed_2, training=training)
#         conv_lvl_4_normed_3 = self.conv_lvl_4_norm_3(conv_lvl_4_out_3, training=training)
#         conv_lvl_4_dropped_1 = self.conv_lvl_4_drop_1(conv_lvl_4_normed_3, training=training)
#         conv_lvl_4_added_1 = self.add_lvl_4([conv_lvl_4_dropped_1, conv_lvl_4_normed_1], training=training)
#         conv_lvl_4_max_1 = self.max_lvl_4(conv_lvl_4_added_1, training=training)

#         # level 5
#         conv_lvl_5_out_1 = self.conv_lvl_5_layer_1(conv_lvl_4_max_1, training=training)
#         conv_lvl_5_normed_1 = self.conv_lvl_5_norm_1(conv_lvl_5_out_1, training=training)
#         conv_lvl_5_out_2 = self.conv_lvl_5_layer_2(conv_lvl_5_normed_1, training=training)
#         conv_lvl_5_normed_2 = self.conv_lvl_5_norm_2(conv_lvl_5_out_2, training=training)
#         conv_lvl_5_out_3 = self.conv_lvl_5_layer_3(conv_lvl_5_normed_2, training=training)
#         conv_lvl_5_normed_3 = self.conv_lvl_5_norm_3(conv_lvl_5_out_3, training=training)
#         conv_lvl_5_dropped_1 = self.conv_lvl_5_drop_1(conv_lvl_5_normed_3, training=training)
#         conv_lvl_5_added_1 = self.add_lvl_5([conv_lvl_5_dropped_1, conv_lvl_5_normed_1], training=training)
#         conv_lvl_5_max_1 = self.max_lvl_5(conv_lvl_5_added_1, training=training)

#         # flatten layer
#         flat_out = self.flat_layer(conv_lvl_5_max_1, training=training)

#         # dense layers
#         dense_out_1 = self.dense_layer_1(flat_out, training=training)
#         dense_normed_1 = self.dense_norm_1(dense_out_1, training=training)
#         dense_dropped_1 = self.dense_drop_1(dense_normed_1, training=training)
#         dense_out_2 = self.dense_layer_2(dense_dropped_1, training=training)
#         dense_normed_2 = self.dense_norm_2(dense_out_2, training=training)
#         dense_dropped_2 = self.dense_drop_2(dense_normed_2, training=training)
#         dense_out_3 = self.dense_layer_3(dense_dropped_2, training=training)

#         return dense_out_3

#     def get_config(self):
#         config = super(LSTM_CNN, self).get_config()
#         config['window_size'] = self.window_size
#         config['n_a'] = self.n_a
#         config['drop_prob'] = self.drop_prob
#         config['filter_size'] = self.filter_size
#         config['kernel_size'] = self.kernel_size

#         return config
    
#     def train_step(self, data):
#         # what if I customize calculation fo loss here in order to add regularization 
#         # and optimize the model based on the regularized loss

#         # unpack data first 
#         X, Y = data

#         # what I want to do is be able to extract the weights of the Dense layer of the SVM
#         # and regularize it
#         with GradientTape() as tape:
#             # pass a batch of X to model representing one forward pass
#             # recall that prediction will be a sigmoid probability
#             Y_pred = self(X, training=True)

#             # compute loss value by using original Y labels
#             # and by using sigmoid probabilities as required by dice loss
#             dice_loss = self.compute_loss(y=Y, y_pred=Y_pred)

#         # Compute gradients
#         coefficients = self.trainable_variables
#         gradients = tape.gradient(dice_loss, coefficients)

#         # Update weights
#         self.optimizer.apply_gradients(zip(gradients, coefficients))

#         # convert probs to solid 1s and 0s for prec and rec metrics
#         pred_train_whole = tf.cast(Y_pred >= 0.2, tf.int64)
#         print(pred_train_whole)

#         # update metrics (includes the metric that tracks the loss)
#         print(f'metrics: {self.metrics}')
#         for metric in self.metrics:
#             print(f'type of metric: {type(metric)}')
#             print(f'metric name: {metric.name}')

#             if metric.name == "loss":
#                 # instead of SquaredHinge in .compile() automatically
#                 # calculating our loss for us which doesn't include regularization
#                 # we pass our manually calculated loss which includes regularization
#                 # in the loss function inside our self.metrics dictionary
#                 metric.update_state(dice_loss)
#             else:
#                 # binary crossentropy, accuracy, f1 score, and auc will
#                 # need sigmoid probabilities as the predictions
#                 metric.update_state(Y, Y_pred)

#         # on the other hand precision and recall metrics need solid 1s
#         # and 0s converted from the sigmoid probabilities
#         self.prec_metric.update_state(Y, pred_train_whole)
#         self.rec_metric.update_state(Y, pred_train_whole)

#         # return a dict mapping metric names to current value
#         return {metric.name: metric.result() for metric in self.metrics + [self.prec_metric, self.rec_metric]}
    
#     # note we do not chnage the inner test_step() mechanism
#     # unlike we did train_step() with regularization this is because
#     # as we recall we should only apply regularization during training
#     # and not in testing
#     def test_step(self, data):
#         # unpack the data
#         X, Y = data

#         # compute predictions given validation data
#         Y_pred = self(X, training=False)

#         # updates the metrics tracking the loss
#         val_dice_loss = self.compute_loss(y=Y, y_pred=Y_pred)

#         # convert probs to solid 1s and 0s for prec and rec metrics
#         pred_test_whole = tf.cast(Y_pred >= 0.2, tf.int64)

#         # update the metrics.
#         for metric in self.metrics:
#             if metric.name != "loss":
#                 metric.update_state(Y, Y_pred)
#             else:
#                 metric.update_state(val_dice_loss)
        
#         self.prec_metric.update_state(Y, pred_test_whole)
#         self.rec_metric.update_state(Y, pred_test_whole)
            
#         # return a dict mapping metric names to current value.
#         # note that it will include the loss (tracked in self.metrics).
#         return {metric.name: metric.result() for metric in self.metrics + [self.prec_metric, self.rec_metric]}

def LSTM_CNN(window_size=5 * 128, n_a=16, drop_prob=0.05, filter_size=32, kernel_size=5, **kwargs):
    # define input size
    inputs = Input((window_size, 1))
    
    # ### LSTM layers ###
    
    lstm1 = LSTM(n_a, activation="tanh", return_sequences=True)(inputs)
    lstm1 = BatchNormalization()(lstm1)
    lstm1_drop = Dropout(drop_prob)(lstm1)

    lstm2 = LSTM(n_a, activation="tanh", return_sequences=True)(lstm1_drop)
    lstm2 = BatchNormalization()(lstm2)
    lstm2_drop = Dropout(drop_prob)(lstm2)

    # ### 1D CNN layers ###
    
    conv1 = Conv1D(filter_size, kernel_size, activation='relu', padding='same',
                   kernel_initializer='he_normal')(lstm2_drop)
    conv1 = BatchNormalization()(conv1)
    conv1_1 = Conv1D(filter_size, kernel_size, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_2 = Conv1D(filter_size, kernel_size, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    drop1 = Dropout(drop_prob)(conv1_2)
    add1 = Add()([drop1, conv1])
    max1 = MaxPooling1D(pool_size=2)(add1)
    
    conv2 = Conv1D(2*filter_size, kernel_size, activation='relu', padding='same',
                   kernel_initializer='he_normal')(max1)
    conv2 = BatchNormalization()(conv2)
    conv2_1 = Conv1D(2*filter_size, kernel_size, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_2 = Conv1D(2*filter_size, kernel_size, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    drop2 = Dropout(drop_prob)(conv2_2)
    add1 = Add()([drop2, conv2])
    max2 = MaxPooling1D(pool_size=2)(add1)
    
    conv3 = Conv1D(4*filter_size, kernel_size, activation='relu', padding='same',
                   kernel_initializer='he_normal')(max2)
    conv3 = BatchNormalization()(conv3)
    conv3_1 = Conv1D(4*filter_size, kernel_size, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_2 = Conv1D(4*filter_size, kernel_size, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    drop3 = Dropout(drop_prob)(conv3_2)
    add3 = Add()([drop3, conv3])
    max3 = MaxPooling1D(pool_size=2)(add3)

    conv4 = Conv1D(8*filter_size, kernel_size, activation='relu', padding='same',
                   kernel_initializer='he_normal')(max3)
    conv4 = BatchNormalization()(conv4)
    conv4_1 = Conv1D(8*filter_size, kernel_size, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_2 = Conv1D(8*filter_size, kernel_size, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4_1)
    conv4_2 = BatchNormalization()(conv4_2)
    drop4 = Dropout(drop_prob)(conv4_2)
    add4 = Add()([drop4, conv4])
    max4 = MaxPooling1D(pool_size=2)(add4)

    conv5 = Conv1D(16*filter_size, kernel_size, activation='relu', padding='same',
                   kernel_initializer='he_normal')(max4)
    conv5 = BatchNormalization()(conv5)
    conv5_1 = Conv1D(16*filter_size, kernel_size, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_2 = Conv1D(16*filter_size, kernel_size, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5_1)
    conv5_2 = BatchNormalization()(conv5_2)
    drop5 = Dropout( drop_prob )(conv5_2)
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
        
    return model
