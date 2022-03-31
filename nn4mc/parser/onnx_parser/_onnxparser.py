from nn4mc.parser._parser import Parser
from nn4mc.datastructures import NeuralNetwork
# from ._layerbuilder import *
# import onnx
# from onnx2keras import onnx_to_keras
# from nn4mc.parser.onnx_parser.onnx_helpers import HDF5Parser
# import numpy as np
from nn4mc.parser.onnx_parser.onnx_helpers import onnx2keras
from nn4mc.parser.hdf5_parser._hdf5parser import HDF5Parser
from tensorflow import keras

class ONNXParser(Parser):

    def __init__(self, file):
        self.file = file
        self.nn = NeuralNetwork()
        self.nn_input_size = None

    def parse(self):
        h5format = onnx2keras(self.file)


        h5parser = HDF5Parser(h5format)
        h5parser.file = self.file
        h5parser.onnx_parse(h5format)
        self.nn = h5parser.nn

    def _parseONNX(self):
        return self.parse.h5format