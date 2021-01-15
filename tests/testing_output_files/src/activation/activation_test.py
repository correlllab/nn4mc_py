import activation
import numpy as np
import unittest
import tensorflow as tf

class ActivationTest(unittest.TestCase):

    def test_sigmoid(self):
        for _ in range(1000):
            x = np.random.uniform(low = -5., high = 5.)
            y_numpy = float(tf.keras.activations.sigmoid(x))
            y_nn4mc = activation.sigmoid(x)
            self.assertTrue(np.isclose(y_nn4mc, y_numpy, 1e-5))
        print("sigmoid tested!")

    def test_softplus(self):
        for _ in range(1000):
            x = np.random.uniform(low = -5., high = 5.)
            y_numpy = float(tf.keras.activations.softplus(x))
            y_nn4mc = activation.softplus(x)
            self.assertTrue(np.isclose(y_nn4mc, y_numpy, 1e-5))
        print("softplus tested")

    def test_softsign(self):
        for _ in range(1000):
            x = np.random.uniform(low = -1000., high = 1000.)
            y_numpy = float(tf.keras.activations.softsign(x))
            y_nn4mc = activation.softsign(x)
            self.assertTrue(np.isclose(y_nn4mc, y_numpy, 1e-5))
        print("softsign tested")

    def test_hard_sigmoid(self):
        for _ in range(1000):
            x = np.random.uniform(low = -1., high = 1.)
            y_numpy = float(tf.keras.activations.hard_sigmoid(x))
            y_nn4mc = activation.hard_sigmoid(x)
            self.assertTrue(np.isclose(y_nn4mc, y_numpy, 1e-5))
        print("hard sigmoid tested")

    def test_relu(self):
        for _ in range(1000):
            x = np.random.uniform(low = -1000., high = 1000.)
            y_numpy = float(tf.keras.activations.relu(x))
            y_nn4mc = activation.relu(x)
            self.assertTrue(np.isclose(y_nn4mc, y_numpy, 1e-5))
        print("relu tested")

    def test_hyper_tan(self):
        for _ in range(1000):
            x = np.random.uniform(low = -1000., high = 1000.)
            y_numpy = float(tf.keras.activations.tanh(x))
            y_nn4mc = activation.hyper_tan(x)
            self.assertTrue(np.isclose(y_nn4mc, y_numpy, 1e-5))
        print("tanh tested")


    def test_softmax(self):
        for _ in range(1000):
            x = np.random.uniform(low = -10., high = 10., size =  10).flatten()
            y_numpy = float(tf.keras.activations.softmax(x))
            y_nn4mc = activation.softmax(x)
            self.assertTrue(np.isclose(y_nn4mc, y_numpy, 1e-5))
        print("softmax tested")

if __name__ == '__main__':
    unittest.main()