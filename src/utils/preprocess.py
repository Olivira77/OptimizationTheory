import numpy as np


def normalize_rows_and_columns(matrix: np.ndarray):
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_rows = matrix / row_sums

    col_sums = matrix.sum(axis=0, keepdims=True)
    normalized_matrix = normalized_rows / col_sums
    print(normalized_matrix.shape)

    return normalized_matrix
