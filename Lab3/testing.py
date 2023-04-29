import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def predict_model(mdl, x, y):
    plt.imshow(x)
    predictions = mdl.predict(np.expand_dims(x, axis=0), verbose=0)
    print('Correct: ', y)
    print('Predicted: ', predictions.argmax())
    plt.title(str(predictions.argmax()))
    plt.show()


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    model = tf.keras.models.load_model('my_model1.h5')
    for (x, y) in zip(x_test[:20], y_test[:20]):
        predict_model(model, x, y)
