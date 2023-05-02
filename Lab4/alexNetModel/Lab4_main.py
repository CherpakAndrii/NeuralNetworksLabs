from matplotlib import pyplot as plt
from get_data import get_data, process_ds, get_ds_size, visualize_data
from nn_model import AlexNet, compile_model
import pandas as pd

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def graph(training_log):
    plt.title('Mean squared error')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(training_log['loss'], label='Train loss')
    plt.plot(training_log['val_loss'], label='Validation loss')
    plt.grid()
    plt.legend()
    plt.show()

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(training_log['accuracy'], label='Train accuracy')
    plt.plot(training_log['val_accuracy'], label='Validation accuracy')
    plt.grid()
    plt.legend()
    plt.show()


def save_log(training_log):
    pd.DataFrame.from_dict(training_log.history).to_csv('log_history.csv')


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
    log_df = pd.DataFrame()

    # model = keras.models.load_model('alexNet2.h5')
    # log_df = pd.read_csv('log_history.csv')

    for i in range(0, 6):
        print('\tphase', i, '\n')
        log = model.fit(train_ds, epochs=5, validation_data=validation_ds, validation_freq=1)
        model.save(f'alexNet{i}.h5')
        log_df = pd.concat([log_df, pd.DataFrame.from_dict(log.history)], ignore_index=True)
        log_df.to_csv('log_history.csv')

    model.evaluate(test_ds)
    graph(log_df)
