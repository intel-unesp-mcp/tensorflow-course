"""
    Exemplify the concept of Tensor

    Set up requirements: 

        conda create -n myenv python=3 numpy scikit-learn tensorflow
        source activate myenv
"""

import numpy as np
import tensorflow as tf

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
m, _ = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

"""
    Rank 0
"""
animal = tf.constant('elephant', name='animal')
number = tf.constant(41, name='number')
pi_val = tf.constant(3.1415, name='pi_val')
print("\nRank 0 tensors")
print(animal)
print(number)
print(pi_val)

"""
    Rank 1
"""
weekend = tf.constant(['saturday', 'sunday'], name='weekend')
digits = tf.constant([1, 2, 3, 4, 5], name='digits')
print("\nRank 1 tensors")
print(weekend)
print(digits)

"""
    Rank 2
"""
ones = tf.constant(np.ones((7, 1)), name='ones')
X = tf.constant(housing_data_plus_bias, name='X')
y = tf.constant(housing.target.reshape(-1, 1), name='y')
print("\nRank 2 tensors")
print(ones)
print(X)
print(y)
