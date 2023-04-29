import tensorflow as tf
from NeuralNetworkModel import get_model, get_learning_rate, compile_model


def train_model(hidden_neurons, x_train, y_train, x_test, y_test, epochs, batch_size):
    mdl = get_model(hidden_neurons)
    lr = get_learning_rate(len(x_train), epochs, batch_size)
    compile_model(mdl, lr)
    mdl.summary()
    mdl.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=1)
    return mdl


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    model = train_model([75, 75, 75], x_train, y_train, x_test, y_test, 30, 120)
    model.evaluate(x_test, y_test)
    model.save('my_model.h5')

