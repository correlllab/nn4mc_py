from context import nnGn, nnDs
import unittest
import numpy as np

class TestGenerator(unittest.TestCase):

    def setUp(self):
        pass

    def test_placeholder(self):
        nn = nnDs.NeuralNetwork()
        layers = []

        layers.append(nnDs.Input('Input Layer'))
        layers.append(nnDs.Conv1D('Conv1D Layer'))
        layers.append(nnDs.Dense('Dense Layer'))

        for layer in layers:
            layer.w.addData(np.array([[0,0],[0,0]]))
            layer.b.addData(np.array([0,0]))

        nn.addLayer(layers[0])
        nn.addLayer(layers[1])
        nn.addLayer(layers[2])

        nn.addEdge(layers[0],layers[1])
        nn.addEdge(layers[1],layers[2])

        generator = nnGn.Generator(nn, 'output')

        generator.generate()


if __name__=='__main__':
    unittest.main()
