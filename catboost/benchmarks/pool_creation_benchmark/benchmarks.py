import numpy as np
import typing as tp
import pandas as pd
from catboost import Pool
from timeit import default_timer


def get_raw_data(rows_count: int, columns_count: int) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    float64_data: np.ndarray = np.float64(np.random.rand(columns_count//4, rows_count))
    float32_data: np.ndarray = np.float32(np.random.rand(columns_count//4, rows_count))
    int8_data: np.ndarray = np.random.randint(np.iinfo(np.int8).min, np.iinfo(np.int8).max,
                                              size=(columns_count//4, rows_count), dtype=np.int8)
    int32_data: np.ndarray = np.random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max,
                                               size=(columns_count//4, rows_count), dtype=np.int32)
    return float64_data, float32_data, int8_data, int32_data


def measure_numpy_ndarray_with_fortran_order_dataset(rows_count: int, columns_count: int = 200) -> tp.Tuple:
    raw_data: tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = get_raw_data(rows_count, columns_count)

    concatenated: np.ndarray = np.transpose(np.concatenate(raw_data, axis=0))
    data: np.ndarray = np.array(concatenated, order='F')

    return measure_pool_building_with_numerical_data(data)


def measure_numpy_ndarray_with_column_order_dataset(rows_count: int, columns_count: int = 200) -> tp.Tuple:
    raw_data: tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = get_raw_data(rows_count, columns_count)

    concatenated: np.ndarray = np.transpose(np.concatenate(raw_data, axis=0))
    data: np.ndarray = np.array(concatenated, order='C')

    return measure_pool_building_with_numerical_data(data)


def measure_numerical_dataframe(rows_count: int, columns_count: int = 200) -> tp.Tuple[float, float]:
    batches: tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = get_raw_data(rows_count, columns_count)

    column_to_data: tp.Dict[str, tp.Any] = {}
    column_prefixes: tp.List[str] = ["float64_", "float32_", "int8_", "int32_"]
    for batch_id in range(len(batches)):
        column_to_data.update({column_prefixes[batch_id] + str(num): batches[batch_id][num]
                               for num in range(batches[batch_id].shape[0])})

    frame = pd.DataFrame(column_to_data)

    return measure_pool_building_with_numerical_data(frame)


def measure_pool_building_with_numerical_data(data: tp.Union[np.ndarray, pd.DataFrame], iteration: int = 10)\
        -> tp.Tuple[float, float]:
    times: tp.List[float] = []
    for _ in range(iteration):
        start: float = default_timer()
        _ = Pool(data)
        finish: float = default_timer()
        times.append(finish - start)
    return np.mean(times), np.std(times)


def get_unique_random_strings(count: int) -> tp.List[str]:
    return ["str_" + str(num) for num in range(count)]


def measure_categorical_dataframe(rows_count: int, columns_count: int = 200, unique_features_count: int = 1000)\
        -> tp.Tuple[float, float]:
    unique_str_values: tp.List[str] = get_unique_random_strings(unique_features_count)
    assert len(set(unique_str_values)) == unique_features_count

    column_data: tp.Dict[str, tp.List[str]] = {}
    column_names: tp.List[str] = ["str_" + str(column_id) for column_id in range(columns_count)]
    for column_id in range(columns_count):
        column_data[column_names[column_id]] = [unique_str_values[value_id]
                                                for value_id in np.random.randint(unique_features_count,
                                                                                  size=rows_count)]

    frame = pd.DataFrame(column_data)

    return measure_pool_building_with_categorical_data(frame, column_names)


def measure_categorical_and_numerical_dataframe(rows_count: int, columns_count: int = 200,
                                                unique_features_count: int = 1000) -> tp.Tuple[float, float]:
    unique_str_values: tp.List[str] = get_unique_random_strings(unique_features_count)
    assert len(set(unique_str_values)) == unique_features_count

    column_data: tp.Dict[str, tp.List[tp.Any]] = {}

    str_columns_names: tp.List[str] = ["str_" + str(num) for num in range(columns_count//2)]
    for column_id in range(columns_count//2):
        values: np.ndarray = np.array([unique_str_values[value_id]
                                       for value_id in np.random.randint(unique_features_count, size=rows_count)])
        column_data[str_columns_names[column_id]] = values

    for column_id in range(columns_count//2):
        column_data["float64_" + str(column_id)] = np.float64(np.random.rand(rows_count))

    frame = pd.DataFrame(column_data)

    return measure_pool_building_with_categorical_data(frame, str_columns_names)


def measure_pool_building_with_categorical_data(data: tp.Union[np.ndarray, pd.DataFrame],
                                                categorical_feature_names: tp.List[str], iteration: int = 10)\
        -> tp.Tuple[float, float]:
    times: tp.List[float] = []
    for _ in range(iteration):
        start: float = default_timer()
        _ = Pool(data, cat_features=categorical_feature_names)
        finish: float = default_timer()
        times.append(finish - start)
    return np.mean(times), np.std(times)


columns: int = 200
rows: int = 100000
print(f"creating catboost.Pool with numpy.ndarray data order=F, rows={rows}, columns={columns}")
measurement: tp.Tuple = measure_numpy_ndarray_with_fortran_order_dataset(rows, columns)
print(f"avg time = {measurement[0]}s, std = {measurement[1]}s\n")

print(f"creating catboost.Pool with numpy.ndarray data order=C, rows={rows}, columns={columns}")
measurement = measure_numpy_ndarray_with_column_order_dataset(rows, columns)
print(f"avg time = {measurement[0]}s, std = {measurement[1]}s\n")

print(f"creating catboost.Pool with pandas.DataFrame with numerical data, rows={rows}, columns={columns}")
measurement = measure_numerical_dataframe(rows, columns)
print(f"avg time = {measurement[0]}s, std = {measurement[1]}s\n")

print(f"creating catboost.Pool with pandas.DataFrame with categorical data, rows={rows}, columns={columns}")
measurement = measure_categorical_dataframe(rows, columns)
print(f"avg time = {measurement[0]}s, std = {measurement[1]}s\n")

print(f"creating catboost.Pool with pandas.DataFrame with categorical and numerical data, rows={rows},"
      f" columns={columns}")
measurement = measure_categorical_and_numerical_dataframe(rows, columns)
print(f"avg time = {measurement[0]}s, std = {measurement[1]}s\n")
