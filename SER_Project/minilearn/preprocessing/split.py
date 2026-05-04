import random
import numpy as np
import math

def test_train_split(*arrays, test_size=None, train_size=None, random_state=None):
    array_length = len(arrays[0])

    for array in arrays:
        if (array_length != len(array)):
            raise RuntimeError("Input arrays must be of same size.")
        
    if (test_size == None):
        if (train_size == None):
            test_size = .25
        else:
            test_size = 1.0 - train_size

    if (train_size == None):
        train_size = 1 - test_size
    
    # allows for consistent results
    rng = np.random.default_rng(random_state)

    # ensures consistent sorting across arrays
    permutation = rng.permutation(array_length)

    split_index = math.floor(array_length * train_size)

    result = []

    for i in range(0, len(arrays)):
        arrays[i] = np.array(arrays[i])
        arrays[i] = arrays[i][permutation]

        result.append(arrays[i][:split_index])
        result.append(arrays[i][split_index:])

    return result