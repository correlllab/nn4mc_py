import nn4mc_py.translator as nnTr
import unittest
import os

class TestTranslator(unittest.TestCase):

    def setUp(self):
        pass

    def test_file(self):
        file = '../data/test_1.hdf5'

        path = os.path.dirname(os.path.abspath(__file__))
        path2 = os.path.join(path, '../output/')
        # print(path2)

        nnTr.translate(file_path=file, file_type='hdf5', output_path=path2)


if __name__=='__main__':
    unittest.main()
