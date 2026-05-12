import numpy as np
import math

def train_test_split(*arrays, test_size=None, train_size=None, random_state=None):
    array_length = len(arrays[0])
        
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

    for array in arrays:
        array = np.array(array)
        array = array[permutation]

        result.append(array[:split_index])
        result.append(array[split_index:])

    return result