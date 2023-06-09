from keras.layers import ReLU, Dense, LSTM, Dropout, BatchNormalization, Conv2D, Reshape, Input, Bidirectional
import tensorflow as tf


def buildModel(input_dim, output_dim, numOfRNN, numOfRNNUnits):
    input = Input((None, input_dim))
    x = Reshape((-1, input_dim, 1))(input)
    x = Conv2D(filters=32, kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=32, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    for i in range(numOfRNN):
        recurrent = LSTM(units=numOfRNNUnits, activation="tanh", recurrent_activation="sigmoid", use_bias=True,
                         return_sequences=True)
        x = Bidirectional(recurrent, merge_mode="concat")(x)
        if i < numOfRNN - 1:
            x = Dropout(rate=0.5)(x)

    x = Dense(numOfRNNUnits * 2)(x)
    x = ReLU()(x)
    x = Dropout(rate=0.5)(x)
    output = Dense(output_dim + 1, activation="softmax")(x)
    model = tf.keras.Model(input, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=CTCLoss)
    return model


def CTCLoss(y_true, y_pred):
    batchLen = tf.cast(tf.shape(y_true)[0], "int64")
    inputLen = tf.cast(tf.shape(y_pred)[1], "int64") * tf.ones(shape=(batchLen, 1), dtype="int64")
    labelLen = tf.cast(tf.shape(y_true)[1], "int64") * tf.ones(shape=(batchLen, 1), dtype="int64")
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, inputLen, labelLen)
    return loss
