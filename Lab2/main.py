import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from neural_network_types import FeedForwardBackprop, CascadeForwardBackprop, ElmanBackprop, NeuralNetworkModel
from training_data_generation import data_split, generate_data


def get_learning_rate(epochs, batch_size):
    initial_learning_rate = 10 ** (-3)
    final_learning_rate = 10 ** (-7)
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / epochs)
    steps_per_epoch = int(len(train) / batch_size)
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=steps_per_epoch,
        decay_rate=learning_rate_decay_factor
    )


def train_model(model_type: type(NeuralNetworkModel), hidden_neurons, train_data, test_data, epochs_num, batch_sz, learning_rate):
    model_t = model_type(hidden_neurons)
    model = model_t.model
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate))
    model.summary()
    return model.fit(np.reshape(train_data[:, :2], (-1, 2)), train_data[:, 2], epochs=epochs_num, batch_size=batch_sz,
                     validation_data=(np.reshape(test_data[:, :2], (-1, 2)), test_data[:, 2]), verbose=1).history, model_t.nn_model_name


def graph(train_loss, val_loss, name):
    plt.title(name+'`s mean_squared_error')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(train_loss, label='Train loss')
    plt.plot(val_loss, label='Validation loss')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data = generate_data(1000000)
    train, test = data_split(data, 0.7)
    epochs = 500
    batch_size = 10000
    lr = get_learning_rate(epochs, batch_size)
    cases = [# (FeedForwardBackprop, [10]), (FeedForwardBackprop, [20]),
             # (CascadeForwardBackprop, [20]), (CascadeForwardBackprop, [10, 10]),
             # (ElmanBackprop, [15]),
        (ElmanBackprop, [10, 10, 10])]

    for (nn_model, hidden_neurons_number) in cases:
        log, name = train_model(nn_model, hidden_neurons_number, train, test, epochs, batch_size, lr)
        graph(log['loss'], log['val_loss'], name)
