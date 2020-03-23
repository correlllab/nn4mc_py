import nn4mc_py.datastructures as nnDs
import unittest

class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        pass

    def test_basic_creation(self):
        pass

    def test_add_layer(self):
        pass

    def test_add_layer_input(self):
        pass

    def test_add_edge(self):
        pass

    def test_add_edge_error_start(self):
        pass

    def test_add_edge_error_end(self):
        pass

    def test_add_edge_error(self):
        pass

    def test_get_layer_exists(self):
        pass

    def test_get_layer_no_exists(self):
        pass

    def test_iterator_no_nodes(self):
        pass

    def test_iterator_no_input(self):
        pass

    def test_iterator_cycle(self):
        pass

    def test_iterator(self):
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
