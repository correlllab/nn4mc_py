
from nn4mc.parser._parser import Parser
from nn4mc.datastructures import NeuralNetwork
from onnx import numpy_helper
from google.protobuf.json_format import MessageToJson,Parse

import onnx
import numpy as np
import json

class OnnxParser(Parser):
    # Maps layer types to code to dynamically build layers
    builder_map = {'Conv1D': 'Conv1DBuilder()',
                   'Conv2D': 'Conv2DBuilder()',
                   'Dense': 'DenseBuilder()',
                   'Flatten': 'FlattenBuilder()',
                   'MaxPooling1D': 'MaxPooling1DBuilder()',
                   'MaxPooling2D': 'MaxPooling2DBuilder()',
                   'Dropout': 'DropoutBuilder()',
                   'SimpleRNN': 'SimpleRNNBuilder()',
                   'GRU': 'GRUBuilder()',
                   'LSTM': 'LSTMBuilder()',
                   'Input': 'InputBuilder()',
                   'Activation': 'ActivationBuilder()'}

    def __init__(self, file):
        self.file = file  # Onnx model file
        self.nn = NeuralNetwork()  # NeuralNetwork to be filled
        self.nn_input_size = None

    def parse(self):
        # Open file if not already file object
        onnxfile = onnx.load(self.file)

        self.parse_model_config(onnxfile)

        graph = onnx.helper.printable_graph(onnxfile.graph)
        print(graph)


    def parse_model_config(self, onnxfile):
        # This function creates the graph and calls the layer builders
        pass

    def parseweight(self, onnxfile):
        INTIALIZERS = onnxfile.graph.initializer
        onnx_weights = {}
        for initializer in INTIALIZERS:
            W = numpy_helper.to_array(initializer)
            onnx_weights[initializer.name] = W



if __name__=='__main__':
    onnx_obj = OnnxParser(file='../../../tests/data/mobilenetv2-7.onnx')
    onnx_obj.parse()
