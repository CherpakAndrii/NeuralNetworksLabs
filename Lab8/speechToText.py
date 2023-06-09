import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from Lab8.CallbackEval import CallbackEval
from nn_model import buildModel


def processSample(label, audio):
    audio = tf.cast(audio, tf.float32)
    spectr = tf.signal.stft(audio, frame_length=256, frame_step=160, fft_length=fft_length)
    spectr = tf.math.pow(tf.math.abs(spectr), 0.5)
    mean = tf.math.reduce_mean(spectr, 1, keepdims=True)
    std = tf.math.reduce_std(spectr, 1, keepdims=True)
    spectr = (spectr - mean) / (std + 1e-10)
    label = charToNum(tf.strings.unicode_split(tf.strings.lower(label), input_encoding='UTF-8'))
    return spectr, label


if __name__ == '__main__':
    trainData, testData = tfds.load("ljspeech", split=["train[:90%]", "train[90%:]"], as_supervised=True)
    characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
    charToNum = tf.keras.layers.StringLookup(vocabulary=characters, oov_token='')
    numToChar = tf.keras.layers.StringLookup(vocabulary=charToNum.get_vocabulary(), oov_token='', invert=True)
    SAMPLE_RATE = 22050
    fft_length = 384

    trainPreprocessed = (trainData.map(processSample, num_parallel_calls=tf.data.AUTOTUNE).padded_batch(32).prefetch(buffer_size=tf.data.AUTOTUNE))
    testPreprocessed = (testData.map(processSample, num_parallel_calls=tf.data.AUTOTUNE).padded_batch(32).prefetch(buffer_size=tf.data.AUTOTUNE))
    for spectrograms, labels in trainPreprocessed.take(1):
        spectrogram = spectrograms[1].numpy()
        spectrogram = np.array([np.trim_zeros(x) for x in np.transpose(spectrogram)])
        label = labels[1]
        label = tf.strings.reduce_join(numToChar(label)).numpy().decode('utf-8')
        plt.imshow(spectrogram, vmax=1)
        print(label)

    model = buildModel(fft_length // 2 + 1, charToNum.vocabulary_size(), 5, 512)
    model.summary()

    callback = CallbackEval(testPreprocessed, model)
    history = model.fit(trainPreprocessed, validation_data=testPreprocessed, epochs=5, callbacks=[callback])
    callback.on_epoch_end(5)

    model.save('/model.h5')
