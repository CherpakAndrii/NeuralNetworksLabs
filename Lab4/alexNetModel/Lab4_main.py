from matplotlib import pyplot as plt
from get_data import get_data, process_ds, get_ds_size, visualize_data
from nn_model import AlexNet, compile_model

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def graph(training_log):
    plt.title('Mean squared error')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(training_log.history['loss'], label='Train loss')
    plt.plot(training_log.history['val_loss'], label='Validation loss')
    plt.grid()
    plt.legend()
    plt.show()

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(training_log.history['accuracy'], label='Train accuracy')
    plt.plot(training_log.history['val_accuracy'], label='Validation accuracy')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_ds, test_ds, validation_ds = get_data()
    visualize_data(train_ds, CLASS_NAMES)

    train_ds_size = get_ds_size(train_ds)
    test_ds_size = get_ds_size(test_ds)
    validation_ds_size = get_ds_size(validation_ds)

    print("Training data size:", train_ds_size)
    print("Test data size:", test_ds_size)
    print("Validation data size:", validation_ds_size)

    train_ds = process_ds(train_ds)
    test_ds = process_ds(test_ds)
    validation_ds = process_ds(validation_ds)

    mdl = AlexNet(len(CLASS_NAMES))
    model = compile_model(mdl)

    log = model.fit(train_ds, epochs=10, validation_data=validation_ds, validation_freq=1)
    model.evaluate(test_ds)

    graph(log)

    model.save('alexNet.h5')
