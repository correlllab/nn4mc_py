import activation
import numpy as np
import unittest
import tensorflow as tf
import ctypes


class ActivationTest(unittest.TestCase):
    def test_sigmoid(self):
        rtol = 1e-5
        size = 10
        for _ in range(1000):
            x = np.random.uniform(low=-1000., high=1000., size = size).flatten()
            y_numpy = float(tf.keras.activations.sigmoid(x))
            y_nn4mc = activation.sigmoid(x, size)
            self.assertTrue(np.isclose(y_nn4mc, y_numpy, rtol))
        print("sigmoid passed!")

    def test_softplus(self):
        rtol = 1e-5
        size = 10
        for _ in range(1000):
            x = np.random.uniform(low = -1000., high = 1000., size = size)
            y_numpy = float(tf.keras.activations.softplus(x))
            y_nn4mc = activation.softplus(x, size)
            self.assertTrue(np.isclose(y_nn4mc, y_numpy, rtol))
        print("softplus passed")

    def test_softsign(self):
        rtol = 1e-5
        size = 10
        for _ in range(1000):
            x = np.random.uniform(low = -1000., high = 1000., size = size)
            y_numpy = float(tf.keras.activations.softsign(x))
            y_nn4mc = activation.softsign(x, size)
            self.assertTrue(np.isclose(y_nn4mc, y_numpy, rtol))
        print("softsign passed")

    def test_hard_sigmoid(self):
        rtol = 1e-5
        size = 10
        for _ in range(1000):
            x = np.random.uniform(low = -1000., high = 1000., size = size)
            y_numpy = np.clip(0.2*x + 0.5, 0., 1.)
            y_nn4mc = activation.hard_sigmoid(x, size)
            self.assertTrue(np.isclose(y_nn4mc, y_numpy, rtol))
        print("hard sigmoid passed")

    def test_relu(self):
        rtol = 1e-5
        size = 10
        for _ in range(1000):
            x = np.random.uniform(low = -1000., high = 1000., size = size).astype(dtype = np.float32)
            y_numpy = float(tf.keras.activations.relu(x))
            y_nn4mc = activation.relu(x, size)
            self.assertTrue(np.isclose(y_nn4mc, y_numpy, rtol))
        print("relu passed")

    def test_hyper_tan(self):
        rtol = 1e-5
        size = 10
        for _ in range(1000):
            x = np.array(np.random.uniform(low = -1000., high = 1000., size = size).tolist(), dtype = np.float).tolist()
            y_numpy = np.tanh(x)
            y_nn4mc = activation.hyper_tan(x, size)
            self.assertTrue(np.isclose(y_nn4mc, y_numpy, rtol))
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
        output_shape = 10
        for _ in range(1000):
            x = np.random.uniform(low = -1000., high = 1000., size =  output_shape).flatten()
            y_numpy = _ref_softmax(x)
            y_nn4mc = activation.softmax(x, output_shape)
            print(x, y_numpy, y_nn4mc)
            self.assertTrue(np.isclose(y_nn4mc, y_numpy, rtol))
        print("softmax passed")

if __name__ == '__main__':
    unittest.main()