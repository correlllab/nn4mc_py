import nn4mc.datastructures as nnDs
import unittest as ut

class TestWeights(ut.TestCase):

    def setUp(self):
        pass

    def test_basic_creation(pass):
        weight = nnDs.Weight('test')

        self.assertEqual('test', weight.id)
        self.assertIsNone(weight.values)

    def test_add_data(self):
        pass

    def test_get_params(self):
        pass

if __name__=='__main__':
    ut.main()
