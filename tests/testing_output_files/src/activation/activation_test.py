import activation
import numpy as np
import unittest
import tensorflow as tf
import ctypes
from typing import List

def swig_py_object_2_list(object, size : int) -> List[float]:
    """
    Converts SwigPyObject to List[float]
    :param object:
    :param size:
    :return:
    """
    y = (ctypes.c_float * size).from_address(int(object))
    new_object = []
    for i in range(size):
        new_object += [y[i]]
    return new_object

def list_2_swig_float_pointer(list : List[float], size : int):
    """
    Converts from list of floats to swig float* object
    :param list:
    :param size:
    :return:
    """
    test_buffer = activation.input(size)
    for i in range(size):
        test_buffer[i] = list[i]
    return test_buffer

class ActivationTest(unittest.TestCase):
    def test_sigmoid(self):
        rtol = 1e-5
        size = 10
        for _ in range(1000):
            x = np.random.uniform(low=-1000., high=1000., size = size).flatten()
            test_buffer = list_2_swig_float_pointer(x.tolist(), size)
            y_numpy = np.array(tf.keras.activations.sigmoid(x)).tolist()
            y_nn4mc = activation.sigmoid(test_buffer.cast(), size)
            y_nn4mc = swig_py_object_2_list(y_nn4mc, size)
            assert np.allclose(y_nn4mc, y_numpy, rtol)
        print("sigmoid passed!")

    def test_softplus(self):
        rtol = 1e-5
        size = 10
        for _ in range(1000):
            x = np.random.uniform(low = -1000., high = 1000., size = size).tolist()
            test_buffer = list_2_swig_float_pointer(x, size)
            y_numpy = np.array(tf.keras.activations.softplus(tf.constant(x, dtype = tf.float32))).tolist()
            y_nn4mc = activation.softplus(test_buffer.cast(), size)
            y_nn4mc = swig_py_object_2_list(y_nn4mc, size)
            assert np.allclose(y_nn4mc, y_numpy, rtol)
        print("softplus passed")

    def test_softsign(self):
        rtol = 1e-5
        size = 10
        for _ in range(1000):
            x = np.random.uniform(low = -1000., high = 1000., size = size).tolist()
            test_buffer = list_2_swig_float_pointer(x, size)
            y_numpy = np.array(tf.keras.activations.softsign(tf.constant(x, dtype = tf.float32))).tolist()
            y_nn4mc = activation.softsign(test_buffer.cast(), size)
            y_nn4mc = swig_py_object_2_list(y_nn4mc, size)
            assert np.allclose(y_nn4mc, y_numpy, rtol)
        print("softsign passed")

    def test_hard_sigmoid(self):
        rtol = 1e-5
        size = 10
        for _ in range(1000):
            x = np.random.uniform(low = -1000., high = 1000., size = size).tolist()
            test_buffer = list_2_swig_float_pointer(x, size)
            y_numpy = np.clip(0.2*np.array(x) + 0.5, 0., 1.)
            y_nn4mc = activation.hard_sigmoid(test_buffer.cast(), size)
            y_nn4mc = swig_py_object_2_list(y_nn4mc, size)
            assert np.allclose(y_nn4mc, y_numpy, rtol = rtol)
        print("hard sigmoid passed")

    def test_relu(self):
        rtol = 1e-5
        size = 10
        for _ in range(1000):
            x = np.random.uniform(low = -1000., high = 1000., size = size).tolist()
            test_buffer = list_2_swig_float_pointer(x, size)
            y_numpy = np.array(tf.keras.activations.relu(tf.constant(x, dtype = tf.float32))).tolist()
            y_nn4mc = activation.relu(test_buffer.cast(), size)
            y_nn4mc = swig_py_object_2_list(y_nn4mc, size)
            assert np.allclose(y_nn4mc, y_numpy, rtol=rtol)
        print("relu passed")

    def test_hyper_tan(self):
        rtol = 1e-5
        size = 10
        for _ in range(1000):
            x = np.random.uniform(low = -1000., high = 1000., size = size).tolist()
            y_numpy = np.tanh(x)
            test_buffer = list_2_swig_float_pointer(x, size)
            y_nn4mc = activation.hyper_tan(test_buffer.cast(), size)
            y_nn4mc = swig_py_object_2_list(y_nn4mc, size)
            assert np.allclose(y_nn4mc, y_numpy, rtol=rtol)
        print("tanh passed")


    def test_softmax(self):
        def _ref_softmax(values):
            """
                Taken from Keras' testing code:
                https://github.com/keras-team/keras/blob/ce5728bbd36004c7a17b86e69a8e59b21d6ee6d4/keras/activations_test.py
            """
            m = np.max(values)
            e = np.exp(values - m)
            return e / np.sum(e)
        rtol = 1e-5
        size = 10
        for _ in range(1000):
            x = np.random.uniform(low = -1000., high = 1000., size = size).flatten()
            y_numpy = _ref_softmax(x)
            test_buffer = list_2_swig_float_pointer(x, size)
            y_nn4mc = activation.softmax(test_buffer.cast(), size)
            y_nn4mc = swig_py_object_2_list(y_nn4mc, size)
            assert np.allclose(y_nn4mc, y_numpy, rtol=rtol)
        print("softmax passed")

if __name__ == '__main__':
    unittest.main()