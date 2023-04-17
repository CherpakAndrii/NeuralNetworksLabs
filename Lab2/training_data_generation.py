from pandas import DataFrame
import numpy as np


def arithmetic_operation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_square = x ** 2
    y_square = y ** 2
    return x_square + y_square


def generate_data(data_size: int, min_val=0, max_val=10) -> DataFrame:
    x = np.random.rand(data_size) * (max_val - min_val) + min_val
    y = np.random.rand(data_size) * (max_val - min_val) + min_val
    result = arithmetic_operation(x, y)
    d = {'x': x, 'y': y, 'result': result}
    df = DataFrame(d)
    return df


def data_split(data: DataFrame, train_percent: float) -> tuple[np.ndarray, np.ndarray]:
    train_idx = int(train_percent * len(data))
    train_data = np.array(data[:train_idx])
    test_data = np.array(data[train_idx:])
    return train_data, test_data
