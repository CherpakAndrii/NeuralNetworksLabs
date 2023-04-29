import tensorflow as tf
from keras.engine.sequential import Sequential
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay


def get_model(neurons_in_hidden_layers: list[int]) -> Sequential:
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(28, 28)))
    model.add(tf.keras.layers.Flatten())
    for neurons in neurons_in_hidden_layers:
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


def get_learning_rate(train_data_length: int, epochs: int, batch_size: int) -> ExponentialDecay:
    initial_learning_rate = 10 ** (-3)
    final_learning_rate = 10 ** (-7)
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / epochs)
    steps_per_epoch = int(train_data_length / batch_size)
    learning_rate = ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=steps_per_epoch,
        decay_rate=learning_rate_decay_factor
    )
    return learning_rate


def compile_model(model, lr):

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
