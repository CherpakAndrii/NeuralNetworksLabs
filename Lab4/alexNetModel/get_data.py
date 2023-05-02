import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.python.data.ops.dataset_ops import DatasetV1
from numpy import ndarray


def get_data() -> tuple[DatasetV1, DatasetV1, DatasetV1]:
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

    validation_images, validation_labels = train_images[:5000], train_labels[:5000]
    train_images, train_labels = train_images[5000:], train_labels[5000:]

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))
    return train_ds, test_ds, validation_ds


def visualize_data(train_ds: DatasetV1, classes: list[str]) -> None:
    plt.figure(figsize=(20, 20))
    for i, (image, label) in enumerate(train_ds.take(25)):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(image)
        plt.title(classes[label.numpy()[0]])
        plt.axis('off')

    plt.show()


def process_images(image: ndarray, label: str) -> tuple[ndarray, str]:
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (227, 227))
    return image, label


def get_ds_size(ds: DatasetV1) -> int:
    return tf.data.experimental.cardinality(ds).numpy()


def process_ds(ds: DatasetV1) -> DatasetV1:
    return ds.map(process_images).shuffle(buffer_size=get_ds_size(ds)).batch(batch_size=32, drop_remainder=True)
