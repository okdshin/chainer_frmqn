import numpy as np

def begin_arange(begin, size):
    return np.arange(begin, begin+size)

def end_arange(end, size):
    return np.arange(end-size, end)

def pad_take(a, indices):
    try:
        sub = np.zeros((len(indices),)+a.shape[1:], dtype=a.dtype)
        for i in range(len(indices)):
            if 0 <= indices[i] and indices[i] < a.shape[0]:
                sub[i] = a[indices[i]]
    except TypeError:
        sub = np.zeros(a.shape[1:], dtype=a.dtype)
        if 0 <= indices and indices < a.shape[0]:
            sub = a[indices]
    return sub

if __name__ == "__main__":
    assert(np.array_equal(begin_arange(begin=0, size=3), [0, 1, 2]))
    assert(np.array_equal(end_arange(end=0, size=3), [-3, -2, -1]))

    assert(np.array_equal(
        pad_take(np.arange(5), begin_arange(begin=0, size=3)),
        [0, 1, 2]))
    assert(np.array_equal(
        pad_take(np.arange(5), begin_arange(begin=-1, size=3)),
        [0, 0, 1]))
    assert(np.array_equal(
        pad_take(np.arange(5), end_arange(end=2, size=3)),
        [0, 0, 1]))
    assert(np.array_equal(
        pad_take(np.arange(5), begin_arange(begin=0, size=10)),
        [0, 1, 2, 3, 4, 0, 0, 0, 0, 0]))
    assert(np.array_equal(
        pad_take(np.arange(5), end_arange(end=5, size=10)),
        [0, 0, 0, 0, 0, 0, 1, 2, 3, 4]))
    assert(pad_take(np.arange(5), 3) == 3)
    assert(pad_take(np.arange(5), -100) == 0)
    assert(pad_take(np.arange(5), 6) == 0)
