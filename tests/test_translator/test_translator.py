import nn4mc.translator as nnTr
import unittest
import os

class TestTranslator(unittest.TestCase):

    def setUp(self):
        pass

    def test_translate(self):
        file = '../data/test_1.hdf5'

        path = os.path.dirname(os.path.abspath(__file__))
        path2 = os.path.join(path, '../output/')
        # print(path2)

        nnTr.translate(file, 'hdf5', path2)

    def test_translate_dict(self):
        with open('../data/test_1.hdf5', 'rb') as file_obj:
            dict = nnTr.translatePlain(file_obj, 'hdf5')

        print(dict)

if __name__=='__main__':
    unittest.main()
