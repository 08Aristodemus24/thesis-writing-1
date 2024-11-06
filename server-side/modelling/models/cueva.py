import tensorflow as tf
from tensorflow import GradientTape
from tensorflow.keras import Sequential, Input, Model 
from tensorflow.keras.losses import SquaredHinge
from tensorflow.keras.metrics import BinaryCrossentropy as bce_metric, BinaryAccuracy, Precision, Recall, AUC, F1Score
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
from tensorflow.keras.regularizers import L2

import numpy as np



@tf.keras.utils.register_keras_serializable()
class GaussianRBF(tf.keras.layers.Layer):
    def __init__(self, units, gamma, **kwargs):
        super(GaussianRBF, self).__init__(**kwargs)
        self.units = units
        self.gamma = tf.keras.backend.cast_to_floatx(gamma)

    def build(self, input_shape):
        # input shape will be based on last hidden state of LSTM which is (m, n_a)
        n_a = input_shape[-1]

        # these are the landmark/centers of the radial basis of shape (n_a, units)
        # function to be subtracted from inputs which is of shape (m, n_a)
        self.mu = self.add_weight(name='mu', shape=(n_a, self.units), initializer='uniform', trainable=True)

    def call(self, inputs):
        # K(x, l) = e^-gamma||x - l||_2 
        # (m, n_a, 1) - m, n_a
        diff = tf.expand_dims(inputs, axis=2) - self.mu

        # compute euclidean distance of the difference 
        # of the inputs to the landmarks/centers
        L2 = tf.reduce_sum(tf.pow(diff, 2), axis=1)
        res = tf.exp(-1 * self.gamma * L2)
        return res

    def compute_output_shape(self, input_shape):
        # first element of input shape represents the number
        # of training examples
        m = input_shape[0]
        
        return (m, self.units)
    
    def get_config(self):
        config = super(GaussianRBF, self).get_config()
        config['units'] = self.units
        config['gamma'] = self.gamma



@tf.keras.utils.register_keras_serializable()
class LSTM_SVM(tf.keras.Model):
    def __init__(self, window_size=5 * 128, n_a=16, drop_prob=0.05, units=10, gamma=0.5, C=1, **kwargs):
        super(LSTM_SVM, self).__init__(**kwargs)
        self.window_size = window_size
        self.n_a = n_a
        self.drop_prob = drop_prob
        self.units = units
        self.gamma = gamma
        self.C = C

        # LSTM layers
        self.lstm_layer_1 = LSTM(units=n_a, activation=tf.nn.tanh, return_sequences=True, name='lstm-layer-1')
        self.lstm_norm_1 = BatchNormalization(name='batch-norm-1')
        self.lstm_drop_1 = Dropout(drop_prob, name='drop-layer-1')

        # whole LSTM has shape (m, 5 * 128, n_a), last hidden state has (m, n_a)
        self.lstm_layer_2 = LSTM(units=n_a, activation=tf.nn.tanh, return_sequences=False, name='lstm-layer-2')
        self.lstm_norm_2 = BatchNormalization(name='batch-norm-2')
        self.lstm_drop_2 = Dropout(drop_prob, name='drop-layer-2')
            
        # SVM layer
        # self.grbf_layer = GaussianRBF(units=units, gamma=gamma, name='gaussian-rbf-layer')
        self.svc_layer = Dense(units=1, activation='linear', name='svc-layer', kernel_regularizer=L2(self.C))

        # extra metrics
        self.prec_metric = Precision(name="precision")
        self.rec_metric = Recall(name="recall")


    def call(self, inputs, training=False):
        # input shape will be a (m, window_size, n_f) (m, Tx, nf) which
        # based on our data we know we can reshape into since we have will
        # have a window size of 5 * 128 according to Llanes-Jurado et al. (2023)
        # so m, 5 * 128, 1 where 1 will only be our number of features since 
        # there will be no feature engineering required

        # LSTM layers
        lstm_out_1 = self.lstm_layer_1(inputs, training=training)
        lstm_normed_1 = self.lstm_norm_1(lstm_out_1, training=training)
        lstm_dropped_1 = self.lstm_drop_1(lstm_normed_1, training=training)

        lstm_out_2 = self.lstm_layer_2(lstm_dropped_1, training=training)
        lstm_normed_2 = self.lstm_norm_2(lstm_out_2, training=training)
        lstm_dropped_2 = self.lstm_drop_2(lstm_normed_2, training=training)

        # SVM layer
        # reduced = self.grbf_layer(lstm_dropped_2)
        # out = self.svc_layer(reduced)

        out = self.svc_layer(lstm_dropped_2)

        return out

    def get_config(self):
        config = super(LSTM_SVM, self).get_config()
        config['window_size'] = self.window_size
        config['n_a'] = self.n_a
        config['drop_prob'] = self.drop_prob
        config['units'] = self.units
        config['gamma'] = self.gamma
        config['C'] = self.C
        

        return config

    def train_step(self, data):
        # what if I customize calculation fo loss here in order to add regularization 
        # and optimize the model based on the regularized loss

        # unpack data first then turn back -1s in Ys to 0s 
        X, Y = data
        Y_true = tf.cast(~(Y == -1), tf.int64)

        # what I want to do is be able to extract the weights of the Dense layer of the SVM
        # and regularize it
        with GradientTape() as tape:
            # pass a batch of X to model representing one forward pass
            Y_pred = self(X, training=True)

            # compute loss value by using original Y labels
            # containing -1s and 1s, as it is needed in squared hinge loss
            sq_hinge_loss = self.compute_loss(y=Y, y_pred=Y_pred)

            # # retrieve weights of Dense/SVC layer and calculate its L2-norm
            # # excluding the bias coefficients of course
            # non_biases = self.trainable_variables[-2]
            # # print(non_biases)
            # reg_term = 0.5 * tf.reduce_sum(tf.square(non_biases))
            # reg_sq_hinge_loss = reg_term + (self.C * sq_hinge_loss)

            # note also that because we calculate a different loss in
            # the train step then by definition calculating the loss given
            # in our compile method would produce a different result since
            # its loss is not regularized

        # Compute gradients
        coefficients = self.trainable_variables
        gradients = tape.gradient(sq_hinge_loss, coefficients)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, coefficients))

        # convert logits/unactivated linear ouptuts to 
        # sigmoid probability values 
        pred_train_probs = tf.nn.sigmoid(Y_pred)
        print(pred_train_probs)

        # convert probs to solid 1s and 0s for prec and rec metrics
        pred_train_whole = tf.cast(pred_train_probs >= 0.2, tf.int64)
        print(pred_train_whole)

        # update metrics (includes the metric that tracks the loss)
        print(f'metrics: {self.metrics}')
        for metric in self.metrics:
            print(f'type of metric: {type(metric)}')
            print(f'metric name: {metric.name}')

            if metric.name == "loss":
                # instead of SquaredHinge in .compile() automatically
                # calculating our loss for us which doesn't include regularization
                # we pass our manually calculated loss which includes regularization
                # in the loss function inside our self.metrics dictionary
                metric.update_state(sq_hinge_loss)
            else:
                # # before passing predicted train labels which are unactivated linear outputs
                # # we need to pass it through tf.sign() first as tf.sign() takes in these values
                # # and outputs -1 if x is < 0, 0 if x is 0, and 1 if x is > 0 which is exactly
                # # what an SVM is supposed to predict
                # pred_train_labels = tf.sign(Y_pred)
                # pred_train_labels = tf.cast(pred_train_labels >= 1, tf.int64)
                metric.update_state(Y_true, pred_train_probs)

        self.prec_metric.update_state(Y_true, pred_train_whole)
        self.rec_metric.update_state(Y_true, pred_train_whole)

        # return a dict mapping metric names to current value
        return {metric.name: metric.result() for metric in self.metrics + [self.prec_metric, self.rec_metric]}
    
    # note we do not chnage the inner test_step() mechanism
    # unlike we did train_step() with regularization this is because
    # as we recall we should only apply regularization during training
    # and not in testing
    def test_step(self, data):
        # unpack the data
        X, Y = data
        Y_true = tf.cast(~(Y == -1), tf.int64)

        # compute predictions given validation data
        Y_pred = self(X, training=False)

        # updates the metrics tracking the loss
        val_sq_hinge_loss = self.compute_loss(y=Y, y_pred=Y_pred)

        # convert logits/unactivated linear ouptuts to 
        # sigmoid probability values 
        pred_test_probs = tf.nn.sigmoid(Y_pred)

        # convert probs to solid 1s and 0s for prec and rec metrics
        pred_test_whole = tf.cast(pred_test_probs >= 0.2, tf.int64)
        print(pred_test_whole)

        # update the metrics.
        for metric in self.metrics:
            if metric.name != "loss":
                
                # perform same transformation of linear values to -1, 0, and 1 to
                # 0 and 1 values
                # pred_test_labels = tf.sign(Y_pred)
                # pred_test_labels = tf.cast(pred_test_labels >= 1, "float")
                metric.update_state(Y_true, pred_test_probs)

            else:
                metric.update_state(val_sq_hinge_loss)
        
        self.prec_metric.update_state(Y_true, pred_test_whole)
        self.rec_metric.update_state(Y_true, pred_test_whole)
            
        # return a dict mapping metric names to current value.
        # note that it will include the loss (tracked in self.metrics).
        return {metric.name: metric.result() for metric in self.metrics + [self.prec_metric, self.rec_metric]}
    


def LSTM_FE(window_size=5 * 128, n_a=16, drop_prob=0.3, lamb_da=0.1, **kwargs):\
    # instantiate sequential model
    model = Sequential()

    # input shape will be (m, 640, 1)
    model.add(Input(shape=(window_size, 1)))

    # input shape will be a (m, window_size, n_f) (m, Tx, nf) which
    # based on our data we know we can reshape into since we have will
    # have a window size of 5 * 128 according to Llanes-Jurado et al. (2023)
    # so m, 5 * 128, 1 where 1 will only be our number of features since 
    # there will be no feature engineering required

    # LSTM layers
    model.add(LSTM(units=n_a, activation=tf.nn.tanh, return_sequences=True, name='lstm-layer-1'))
    model.add(Dropout(drop_prob, name='lstm-drop-1'))

    # whole LSTM has shape (m, 5 * 128, n_a), last hidden state has (m, n_a)
    model.add(LSTM(units=n_a, activation=tf.nn.tanh, return_sequences=False, name='lstm-layer-2'))
    model.add(Dropout(drop_prob, name='lstm-drop-2'))
        
    # dense layer
    model.add(Dense(units=10, name='dense-layer-1'))
    model.add(BatchNormalization(name='dense-norm-1'))
    model.add(Activation(activation=tf.nn.relu, name='act-layer-1'))
    model.add(Dropout(drop_prob, name='dense-drop-1'))

    model.add(Dense(units=3, name='dense-layer-2'))
    model.add(BatchNormalization(name='dense-norm-2'))
    model.add(Activation(activation=tf.nn.relu, name='act-layer-2'))

    model.add(Dense(units=1, activation=tf.nn.sigmoid, name='out-layer', kernel_regularizer=L2(lamb_da)))

    return model