from context import nnPr
import unittest
from nn4mc_py.parser.hdf5_parser._layerbuilder import *

class TestHDF5Parser(unittest.TestCase):

    def setUp(self):
        pass

    def test_basic_1(self):
        p = nnPr.HDF5Parser('Test')

        print('OK')

    def test_file_1(self):
        p = nnPr.HDF5Parser('../data/test_1.hdf5')

        p.parse()

        for node in p.nn.iterate():
            print(node.layer.identifier)
            print(node.layer.w.values)


if __name__=='__main__':
    unittest.main()
