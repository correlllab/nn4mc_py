from context import nnPr
import unittest
from nn4mc_py.parser.hdf5_parser._layerbuilder import *

class TestHDF5Parser(unittest.TestCase):

    def setUp(self):
        pass

    def test_basic_1(self):
        k = nnPr.HDF5Parser('Test')
        str = k.builder_map['Dense']
        b = eval(str)

        print(type(b))


if __name__=='__main__':
    unittest.main()
