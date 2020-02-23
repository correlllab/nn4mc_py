from context import nnPr, nnDs, nnGn
import unittest
import os

class TestTranslator(unittest.TestCase):

    def setUp(self):
        pass

    def test_file(self):
        p = nnPr.HDF5Parser('../data/test_1.hdf5')

        p.parse()

        path = os.path.dirname(os.path.abspath(__file__))
        path2 = os.path.join(path, 'output/')
        print(path2)

        generator = nnGn.Generator(p.nn, path2)

        generator.generate()


if __name__=='__main__':
    unittest.main()
