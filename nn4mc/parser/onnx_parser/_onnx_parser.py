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
        'conv' : ConvBuilder,
        'gemm' : DenseBuilder,
        'maxpool' : MaxPoolBuilder,
        'rnn' : SimpleRNNBuilder,
        'gru' : GRUBuilder,
        'lstm' : LSTMBuilder,
        'activation' : ActivationBuilder,
        'dropout' : DropoutBuilder,
        'input' : InputBuilder,
        'flatten' : FlattenBuilder
    }
    activations = ['sigmoid', 'softplus', 'softsign', 'hardsigmoid',
                    'exp', 'relu', 'tanh', 'softmax']

    def __init__(self, file):
        self.file = file
        self.nn = NetworkGraph()

    def parse(self):
        onnx_model = onnx.load(file)

        #Can check model ir_version or opset_import here

        self.parseModelConfig(onnx_model)

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
            type = node.op_type.lower()
            name = (node.name).lower()

            try:
                builder = self.builder_map[type]()
                weights = self.parseWeight(node.inputs, params)

                layer = builder.build_layer(node.attribute, weights, input_shape, name)
            except:
                if type in self.activations:
                    builder = self.builder_map['activation']()
                    layer = builder.build_layer(name, type)
                else:
                    raise ValueError('Unsupported layer type encountered.')

            self.nn.addLayer(layer)
            self.nn.addEdge(last_layer, layer)

            input_shape = layer.computeOutShape()
            last_layer = layer

    def parseWeight(self, inputs, params):
        weight_inputs = []
        num_params = 0
        for input in inputs:
            if 'bias' in input or 'weight' in input:
                weight_inputs.append(input)
                num_params += 1

        if num_params == 0:
            return None

        weights = {}

        for param in params:
            if param.name in inputs:
                num_params -= 1

                key = param.name.split('.')[1]
                weights[key] = to_array(param)

            if num_params == 0:
                break

        return weights

    def parseInput(self, model_input):
        #This assumes there is only one input
        input_shape = model_input[0].type.tensor_type.shape

        #Here we are assuming the first dim is the batch dim which we ignore
        return [dim.dim_value for dim in shape.dim[1:]]
