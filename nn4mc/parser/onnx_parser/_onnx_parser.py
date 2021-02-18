from nn4mc.datastructures import NetworkGraph
from ._layer_builder import *
import onnx
import onnx.numpy_helper.to_array as to_array
import numpy as np

'''
This class implements functionality to parse an neural network model saved
in the ONNX file format.
'''
class ONNXParser():
    #Maps layer types to builder handles
    builder_map = {
        'Conv' : ConvBuilder,
        'Gemm' : DenseBuilder,
        'MaxPool' : MaxPoolBuilder,
        'RNN' : SimpleRNNBuilder,
        'GRU' : GRUBuilder,
        'LSTM' : LSTMBuilder,
        'Activation' : ActivationBuilder,
        'Dropout' : DropoutBuilder,
        'Input' : InputBuilder,
        'Flatten' : FlattenBuilder
    }

    def __init__(self, file):
        self.file = file
        self.nn = NetworkGraph()

    def parse(self):
        onnx_model = onnx.load(file)

        #Can check model ir_version or opset_import here

        self.parseModelConfig(onnx_model)

        self.parseWeights(onnx_model)

    def parseModelConfig(self, onnx_model):
        graph = onnx_model.node
        params = onnx_model.initializer

        #Deal with input dimensions
        input_shape = self.parse_nn_input(onnx_model.input)

        #NOTE: We are assuming the model is sequential
        print('Assuming model is sequential.')

        last_layer = Input('input_1', 'input')
        self.nn.addLayer(last_layer)

        for node in graph:
            type = node.op_type
            name = node.name
            builder = self.builder_map[type]()

            weights = self.parseWeight(node.inputs, params)

            layer, output_shape = builder.build_layer(node.attribute,
                                                        weights,
                                                        input_shape,
                                                        name.lower(),
                                                        type.lower())

            self.nn.addLayer(layer)
            self.nn.addEdge(last_layer, layer)

            input_shape = output_shape
            last_layer = layer

    def parseWeight(self, inputs, params):
        weights = {}

        for param in params:
            if param.name in inputs:
                key = param.name.split('.')[1]
                weights[key] = to_array(param)

        return weights

    def parseInput(self, model_input):
        #This assumes there is only one input
        input_shape = model_input[0].type.tensor_type.shape

        #Here we are assuming the first dim is the batch dim which we ignore
        return [dim.dim_value for dim in shape.dim[1:]]
