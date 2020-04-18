import nn4mc.parser as nnPr
import nn4mc.datastructures as nnDs
import nn4mc.generator as nnGn
import unittest
import os

class TestTranslator(unittest.TestCase):

    def setUp(self):
        pass

    def test_file(self):
        p = nnPr.HDF5Parser('../data/test_1.hdf5')

        p.parse()

        path = os.path.dirname(os.path.abspath(__file__))
        path2 = os.path.join(path, '../output/')
        # print(path2)

        generator = nnGn.Generator(p.nn)

        generator.generate(path2)


if __name__=='__main__':
    unittest.main()
