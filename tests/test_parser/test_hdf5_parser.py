import nn4mc.parser as nnPr
import unittest
import io

class TestHDF5Parser(unittest.TestCase):

    def setUp(self):
        pass

    def test_basic_creation(self):
        parser = nnPr.HDF5Parser('test')

        self.assertEqual('test', parser.file)
        # self.assertIsInstance(NeuralNetwork, parser.nn)
        self.assertIsNone(parser.nn_input_size)

    def test_file_object_1(self):
        with open('../data/test_1.hdf5', 'rb') as file_obj:
            p = nnPr.HDF5Parser(file_obj)
            p.parse()

            for node in p.nn.iterate():
                print(node.layer.identifier)

    # def test_file_object_2(self):
    #     with open('../data/test_1.hdf5', 'rb') as file:
    #         data = file.read()
    #
    #     data = data.decode('utf-8')
    #
    #     byt = data.encode('utf-8')
    #     f = io.BytesIO(byt)
    #     p = nnPr.HDF5Parser(f)
    #     p.parse()
    #
    #     for node in p.nn.iterate():
    #         print(node.layer.identifier)

    # def test_file_1(self):
    #     p = nnPr.HDF5Parser('../data/test_1.hdf5')
    #
    #     p.parse()
    #
    #     for node in p.nn.iterate():
    #         if node.layer.w is not None:
    #             print(node.layer.identifier)
    #             print(node.layer.w.values)


if __name__=='__main__':
    unittest.main()
