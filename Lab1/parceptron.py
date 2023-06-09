import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    x_train = np.array([[a, b, c] for a in range(2) for b in range(2) for c in range(2)], 'float32')
    y_train = np.array([[sum(arr) % 2] for arr in x_train])

    print(x_train)
    print(y_train)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(3))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=50)
    scores = model.evaluate(x_train, y_train, verbose=False)
    print('\n%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))

    print(model.predict(x_train).round())
