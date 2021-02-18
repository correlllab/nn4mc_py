from nn4mc.datastructures import NetworkGraph
from ._layer_builder import *
import onnx
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
        self.nn_input_shape = None

    def parse(self):
        onnx_model = onnx.load(file)

        #Can check model ir_version or opset_import here

        self.parseModelConfig(onnx_model)

        self.parseWeights(onnx_model)

    def parseModelConfig(self, onnx_model):
        graph = onnx_model.node
        params = onnx_model.initializers

        #Deal with input dimensions
        self.parse_nn_input(onnx_model.input)

        #NOTE: We are assuming the model is sequential
        print('Assuming model is sequential.')

        last_layer = Input('input_1', 'input')
        self.nn.addLayer(last_layer)

        for node in graph:
            type = node.op_type
            name = node.name
            builder = self.builder_map[type]()

            layer = builder.build_layer(node.attribute, name.lower(), type.lower())

            weights = self.parseWeight(node.inputs)
            ids = [] #Something here
            layer.addParameters(weights, ids)

            self.nn.addLayer(layer)
            self.nn.addEdge(last_layer, layer)

            last_layer = layer

    def parseWeight(self, inputs):
        pass

    def parseInput(self, model_input):
        #This assumes there is only one input
        input_shape = model_input[0].type.tensor_type.shape

        #Here we are assuming the first dim is the batch dim which we ignore
        self.nn_input_shape = [dim.dim_value for dim in shape.dim[1:]]
