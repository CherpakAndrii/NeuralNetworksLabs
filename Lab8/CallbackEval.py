import tensorflow as tf
import numpy as np
from jiwer import wer

characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
charToNum = tf.keras.layers.StringLookup(vocabulary=characters, oov_token='')
numToChar = tf.keras.layers.StringLookup(vocabulary=charToNum.get_vocabulary(), oov_token='', invert=True)


def decodePredictions(pred):
    inputLen = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=inputLen, greedy=True)[0][0]
    outputText = [tf.strings.reduce_join(numToChar(result)).numpy().decode('utf-8') for result in results]
    return outputText


class CallbackEval(tf.keras.callbacks.Callback):
    def __init__(self, dataset, model):
        super().__init__()
        self.dataset = dataset
        self.model = model

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for spectrograms, labels in self.dataset:
            batch_predictions = self.model.predict(spectrograms, verbose=0)
            batch_predictions = decodePredictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in labels:
                label = (tf.strings.reduce_join(numToChar(label)).numpy().decode('utf-8'))
                targets.append(label)
            wer_score = wer(targets, predictions)
        print('-' * 100)
        print(f'Word Error Rate: {wer_score:.4f}')
        print('-' * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f'Target : {targets[i]}')
            print(f'Prediction: {predictions[i]}')
            print('-' * 100)
