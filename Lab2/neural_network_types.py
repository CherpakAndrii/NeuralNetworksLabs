import tensorflow as tf
from keras import Model
from keras.layers import Dense, Input, Concatenate, SimpleRNN


class NeuralNetworkModel:
    def __init__(self, name: str, model: Model):
        self.nn_model_name = name
        self.model = model


class FeedForwardBackprop(NeuralNetworkModel):
    def __init__(self, hidden_neuron_numbers: list[int]):
        neurons_str_repr = f"{hidden_neuron_numbers[0]}"
        for hnn in hidden_neuron_numbers[1:]:
            neurons_str_repr += f", {hnn}"
        model_name = f"FeedForwardBackprop({neurons_str_repr})"

        input_layer = Input(2)
        current_layer = input_layer
        for neurons in hidden_neuron_numbers:
            current_layer = Dense(neurons, activation='relu')(current_layer)
        output_layer = Dense(1, activation='relu')(current_layer)

        model = Model(input_layer, output_layer)
        super().__init__(model_name, model)


class CascadeForwardBackprop(NeuralNetworkModel):
    def __init__(self, hidden_neuron_numbers: list[int]):
        neurons_str_repr = f"{hidden_neuron_numbers[0]}"
        for hnn in hidden_neuron_numbers[1:]:
            neurons_str_repr += f", {hnn}"
        model_name = f"CascadeForwardBackprop({neurons_str_repr})"

        input_layer = Input(2)
        concatenated_layers = input_layer
        for neuron_number in hidden_neuron_numbers:
            hidden_layer = Dense(neuron_number, activation='relu')(concatenated_layers)
            concatenated_layers = Concatenate(axis=-1)([concatenated_layers, hidden_layer])
        output_layer = Dense(1, activation='relu')(concatenated_layers)

        model = Model(input_layer, output_layer)
        super().__init__(model_name, model)


class ElmanBackprop(NeuralNetworkModel):
    def __init__(self, hidden_neuron_numbers: list[int]):
        neurons_str_repr = f"{hidden_neuron_numbers[0]}"
        for hnn in hidden_neuron_numbers[1:]:
            neurons_str_repr += f", {hnn}"
        model_name = f"ElmanBackprop({neurons_str_repr})"

        input_layer = Input(2)
        current_layer = tf.expand_dims(input_layer, axis=1)
        current_layer = SimpleRNN(hidden_neuron_numbers[0])(current_layer)
        for neuron_number in hidden_neuron_numbers[1:]:
            current_layer = tf.expand_dims(current_layer, axis=1)
            current_layer = SimpleRNN(neuron_number, activation='relu')(current_layer)
        output_layer = Dense(1, activation='relu')(current_layer)

        model = Model(input_layer, output_layer)
        super().__init__(model_name, model)
