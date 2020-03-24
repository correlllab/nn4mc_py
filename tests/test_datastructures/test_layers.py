import nn4mc_py.datastructures as nnDs
import unittest as ut

class TestLayers(ut.TestCase):

    def setUp(self):
        pass

    def test_basic_creation(self):
        layer = nnDs.Layer('test')

        self.assertEqual('test', layer.id)
        self.assertEqual('unspecified', layer.layer_type)

    def test_add_parameters(self):
        pass

    #Not sure what to test here as there are a lot of derived classes

if __name__ == "__main__":
    ut.main()
