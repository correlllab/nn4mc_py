from context import nnDs
import unittest

class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        pass

    def test_iterator_1(self):
        nn = nnDs.NeuralNetwork()
        layers = []

        layers.append(nnDs.Input('Input Layer'))
        layers.append(nnDs.Conv1D('Conv1D Layer'))
        layers.append(nnDs.Dense('Dense Layer'))

        nn.addLayer(layers[0])
        nn.addLayer(layers[1])
        nn.addLayer(layers[2])

        nn.addEdge(layers[0],layers[1])
        nn.addEdge(layers[1],layers[2])

        for node in nn.iterate():
            print(node.layer.identifier)


if __name__ == "__main__":
    unittest.main()
