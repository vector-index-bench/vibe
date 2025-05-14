from typing import Callable, NamedTuple

import numba
import numpy as np
from numpy.linalg import norm


def euclidean(a, b):
    if a.dtype == np.uint8:
        a = a.astype(np.int16)
    if b.dtype == np.uint8:
        b = b.astype(np.int16)

    if a.ndim == 1 and b.ndim == 1:
        return norm(a - b)
    if a.ndim == 1 and b.ndim == 2:
        return norm(b - a, axis=1)
    if a.ndim == 2 and b.ndim == 1:
        return norm(a - b, axis=1)
    A = np.sum(a * a, axis=1)[:, None]
    B = np.sum(b * b, axis=1)[None, :]
    return np.sqrt(np.maximum(A + B - 2 * np.dot(a, b.T), 0))


def cosine(a, b):
    # lacks handling for zero vectors as datasets should never have them
    if a.ndim == 1 and b.ndim == 1:
        return np.clip(1 - np.dot(a, b) / (norm(a) * norm(b)), a_min=0, a_max=2)
    if a.ndim == 1 and b.ndim == 2:
        return np.clip(1 - np.dot(b, a) / (norm(a) * norm(b, axis=1)), a_min=0, a_max=2)
    if a.ndim == 2 and b.ndim == 1:
        return np.clip(1 - np.dot(a, b) / (norm(a, axis=1) * norm(b)), a_min=0, a_max=2)
    norm_a = norm(a, axis=1)
    norm_b = norm(b, axis=1)
    return np.clip(1 - np.dot(a, b.T) / (norm_a[:, None] * norm_b[None, :]), a_min=0, a_max=2)


def ip(a, b):
    if a.ndim == 1 and b.ndim == 1:
        return -np.dot(a, b)
    if a.ndim == 1 and b.ndim == 2:
        return -np.dot(b, a)
    if a.ndim == 2 and b.ndim == 1:
        return -np.dot(a, b)
    return -np.dot(a, b.T)


def normalized_cosine(a, b):
    # cosine distance assuming a and b are already normalized
    return np.clip(1 + ip(a, b), a_min=0, a_max=2)


popcnt = np.array([bin(i).count("1") for i in range(256)], dtype=np.float32)


@numba.njit(
    [
        "f4(u1[::1],u1[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.uint8, 1, "C", readonly=True),
            numba.types.Array(numba.types.uint8, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "intersection": numba.types.uint8,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def bit_hamming(x, y):
    result = 0.0
    dim = x.shape[0]

    for i in range(dim):
        intersection = x[i] ^ y[i]
        result += popcnt[intersection]

    return result


def hamming(a, b):
    if a.ndim == 1 and b.ndim == 1:
        return bit_hamming(a, b)
    if a.ndim == 1 and b.ndim == 2:
        return np.array([bit_hamming(a, x) for x in b])
    if a.ndim == 2 and b.ndim == 1:
        return np.array([bit_hamming(x, b) for x in a])
    return np.array([[bit_hamming(x, y) for y in b] for x in a])


class Metric(NamedTuple):
    distance: Callable[[np.ndarray, np.ndarray], float]


metrics = {
    "euclidean": Metric(distance=euclidean),
    "cosine": Metric(distance=cosine),
    "ip": Metric(distance=ip),
    "normalized": Metric(distance=normalized_cosine),
    "hamming": Metric(distance=hamming),
}
