from context import nnGn, nnDs
import unittest
import numpy as np
import os

class TestGenerator(unittest.TestCase):

    def setUp(self):
        pass

    def test_basic(self):
        nn = nnDs.NeuralNetwork()
        layers = []

        layers.append(nnDs.Input('Input Layer', 'input'))
        layers.append(nnDs.Dense('dense_1', 'dense'))
        layers[0].w.identifier = 'w1'
        layers[0].b.identifier = 'b1'
        layers.append(nnDs.Dense('dense_2', 'dense'))
        layers[1].w.identifier = 'w2'
        layers[1].b.identifier = 'b2'

        for layer in layers:
            if layer.layer_type != 'Input':
                layer.w.addData(np.array([[0,0],[0,0]]))
                layer.b.addData(np.array([0,0]))

                layer.input_shape[0] = 2
                layer.output_size = 2
                layer.activation = 'sigmoid'

        nn.addLayer(layers[0])
        nn.addLayer(layers[1])
        nn.addLayer(layers[2])

        nn.addEdge(layers[0], layers[1])
        nn.addEdge(layers[1], layers[2])

        for node in nn.iterate():
            print(node.layer.identifier)

        path = os.path.dirname(os.path.abspath(__file__))
        path2 = os.path.join(path, 'output/')
        print(path2)

        generator = nnGn.Generator(nn, path2)

        generator.generate()


if __name__=='__main__':
    unittest.main()
