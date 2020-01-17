from context import nnAl
import unittest

class TestAnalyzer(unittest.TestCase):

    def setUp():
        pass

    def test_placeholder():
        AL = nnAl.Analyzer()

        AL.load_model(sys.argv[0])
        AL.cheapify()
        AL.compare()


if __name__ == '__main__':
    unittest.main()
