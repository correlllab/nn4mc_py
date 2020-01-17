from .context import nn4mp_py.datastructures as nnDs
import unittest

def TestNeuralNetwork(unittest.TestCase):
    
    def setUp():
        pass
        
    def test_iterator_1():
        nn = nnDs.NeuralNetwork()

        l0 = nnDs.Input('Input Layer')
        l1 = nnDs.Conv1D('Conv1D Layer')
        l2 = nnDs.Dense('Dense Layer')

        nn.addLayer(l0)
        nn.addLayer(l1)
        nn.addLayer(l2)
        nn.addLayer(l3)

        nn.addEdge(l0,l1)
        nn.addEdge(l1,l2)

        for node in nn.iterate():
            print(node.layer.identifier)


if __name__ == "__main__":
    unittest.main()
