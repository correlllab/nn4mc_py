from nn4mc.datastructures.layer._layer import *
from abc import ABC, abstractmethod

#Parent class for all layer builders
class LayerBuilder(ABC):
    weight_map = {
        'weight' : '_W',
        'bias' : '_b',
        'weight_rec' : '_Wrec'
    }

    #Builds a layer object from JSON metadata
    @abstractmethod
    def build_layer(self, attributes, weights, input_shape, id):
        pass

################################################################################
#Derived classes
'''
ONNX operators
https://github.com/onnx/onnx/blob/master/docs/Operators.md
'''

class ConvBuilder(LayerBuilder):
    def build_layer(self, attributes, weights, input_shape, id):

        #Need to figure out which dimension conv
        config = {item.name : item.ints for item in attributes}

        if len(config['kernel_shape']) == 1:
            new_layer = Conv1D(id)
        elif len(config['kernel_shape']) == 2:
            new_layer = Conv2D(id)
        else: raise ValueError('Only 1D and 2D convolutions supported.')

        new_layer.filters = weights['weight'].shape[0]
        new_layer.kernel_shape = config['kernel_shape']
        new_layer.strides = config['strides']

        if 'auto_pad' in config.keys():
            new_layer.padding = config['auto_pad'].split('_')[0].lower()
        elif all(p == 0 for p in config['pads']):
            new_layer.padding = 'valid'
        else: raise ValueError('Custom padding not supported.')

        new_layer.data_format = 'channels_first'
        new_layer.dilation_rate = config['dilations']

        #Deal with weights
        for key, value in weights.items():
            new_layer.addParameters(key, (id+self.weight_map[key], value))

        return new_layer

class DenseBuilder(LayerBuilder):
    def build_layer(self, attributes, weights, input_shape, id):
        new_layer = Dense(id)

        new_layer.units = weights['weight'].shape[0]

        for key, value in weights.items():
            new_layer.addParameters(key, (id+self.weight_map[key], value))

        return new_layer

class MaxPoolingBuilder(LayerBuilder):
    def build_layer(self, attributes, weights, input_shape, id):

        config = {item.name : item.ints for item in attributes}

        if len(config['kernel_shape']) == 1:
            new_layer = MaxPooling1D(id)
        elif len(config['kernel_shape']) == 2:
            new_layer = MaxPooling2D(id)
        else: raise ValueError('Only 1D and 2D maxpooling supported.')

        new_layer.pool_shape = config['kernel_shape']
        new_layer.strides = config['strides']

        if 'auto_pad' in config.keys():
            new_layer.padding = config['auto_pad'].split('_')[0].lower()
        elif all(p == 0 for p in config['pads']):
            new_layer.padding = 'valid'
        else: raise ValueError('Custom padding not supported.')

        new_layer.data_format = 'channels_first'

        return new_layer

class SimpleRNNBuilder(LayerBuilder):
    def build_layer(self, attributes, weights, input_shape, id):
        new_layer = SimpleRNN(id)

        # new_layer.units =
        # new_layer.activation =
        # new_layer.use_bias =
        # new_layer.return_sequences =
        # new_layer.return_state =
        # new_layer.go_backwards =
        # new_layer.stateful =

        return new_layer

class GRUBuilder(LayerBuilder):
    def build_layer(self, attributes, weights, input_shape, id):
        new_layer = GRU(id)

        # new_layer.units =
        # new_layer.dropout =
        # new_layer.recurrent_dropout =
        # new_layer.activation =
        # new_layer.recurrent_activation =
        # new_layer.use_bias =
        # new_layer.go_backwards =
        # new_layer.stateful =
        # new_layer.unroll =
        # new_layer.reset_after =

        return new_layer

class LSTMBuilder(LayerBuilder):
    def build_layer(self, attributes, weights, input_shape, id):
        new_layer = LSTM(id)

        # new_layer.units =
        # new_layer.dropout =
        # new_layer.recurrent_dropout =
        # new_layer.activation =
        # new_layer.recurrent_activation =
        # new_layer.implementation =
        # new_layer.use_bias =
        # new_layer.go_backwards =
        # new_layer.stateful =
        # new_layer.unroll =

        return new_layer

class ActivationBuilder(LayerBuilder):
    def build_layer(self, id, type):
        new_layer = Activation(id)

        new_layer.activation = type

        return new_layer

class DropoutBuilder(LayerBuilder):
    def build_layer(self, attributes, weights, input_shape, id):
        new_layer = Dropout(id)

        return new_layer

class FlattenBuilder(LayerBuilder):
    def build_layer(self, attributes, weights, input_shape, id):
        new_layer = Flatten(id)

        return new_layer

class InputBuilder(LayerBuilder):
    def build_layer(self, attributes, weights, input_shape, id):
        new_layer = Input(id)

        return new_layer
