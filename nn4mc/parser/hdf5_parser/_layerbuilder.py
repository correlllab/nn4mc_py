from nn4mc.datastructures.layer._layer import *
from nn4mc.generator.code_generator._globals import activation_lookup
from abc import ABC, abstractmethod
import copy

#Parent class for all layer builders
class LayerBuilder(ABC):

    #Builds a layer object from JSON metadata
    @abstractmethod
    def build_layer(self, json_obj, id, layer_type):
        pass

################################################################################
#Derived classes

class Conv1DBuilder(LayerBuilder):
    def build_layer(self, json_obj, id, layer_type):
        new_layer = Conv1D(id, layer_type)

        new_layer.filters = json_obj['filters']
        new_layer.kernel_shape = copy.copy(json_obj['kernel_size'])
        new_layer.strides = copy.copy(json_obj['strides'])
        new_layer.padding = json_obj['padding']
        new_layer.data_format = json_obj['data_format']
        new_layer.dilation_rate = copy.copy(json_obj['dilation_rate'])
        new_layer.activation = activation_lookup[json_obj['activation']]
        new_layer.use_bias = json_obj['use_bias']

        return new_layer

class Conv2DBuilder(LayerBuilder):
    def build_layer(self, json_obj, id, layer_type):
        new_layer = Conv2D(id, layer_type)

        new_layer.filters = json_obj['filters']
        new_layer.kernel_shape = copy.copy(json_obj['kernel_size'])
        new_layer.strides = copy.copy(json_obj['strides'])
        new_layer.padding = json_obj['padding']
        new_layer.data_format = json_obj['data_format']
        new_layer.dilation_rate = copy.copy(json_obj['dilation_rate'])
        new_layer.activation = activation_lookup[json_obj['activation']]
        new_layer.use_bias = json_obj['use_bias']

        return new_layer

class DenseBuilder(LayerBuilder):
    def build_layer(self, json_obj, id, layer_type):
        new_layer = Dense(id, layer_type)

        new_layer.units = json_obj['units']
        new_layer.activation = activation_lookup[json_obj['activation']]
        new_layer.use_bias = json_obj['use_bias']

        return new_layer

class MaxPooling1DBuilder(LayerBuilder):
    def build_layer(self, json_obj, id, layer_type):
        new_layer = MaxPooling1D(id, layer_type)

        new_layer.pool_shape = copy.copy(json_obj['pool_size'])
        new_layer.strides = copy.copy(json_obj['strides'])
        new_layer.padding = json_obj['padding']
        new_layer.data_format = json_obj['data_format']

        return new_layer

class MaxPooling2DBuilder(LayerBuilder):
    def build_layer(self, json_obj, id, layer_type):
        new_layer = MaxPooling2D(id, layer_type)

        new_layer.pool_shape = copy.copy(json_obj['pool_size'])
        new_layer.strides = copy.copy(json_obj['strides'])
        new_layer.padding = json_obj['padding']
        new_layer.data_format = json_obj['data_format']

        return new_layer

class SimpleRNNBuilder(LayerBuilder):
    def build_layer(self, json_obj, id, layer_type):
        new_layer = SimpleRNN(id, layer_type)

        new_layer.units = json_obj['units']
        new_layer.activation = activation_lookup[json_obj['activation']]
        new_layer.use_bias = json_obj['use_bias']
        new_layer.return_sequences = json_obj['return_sequences']
        new_layer.return_state = json_obj['return_state']
        new_layer.go_backwards = json_obj['go_backwards']
        new_layer.stateful = json_obj['stateful']

        return new_layer

class GRUBuilder(LayerBuilder):
    def build_layer(self, json_obj, id, layer_type):
        new_layer = GRU(id, layer_type)

        new_layer.units = json_obj['units']
        new_layer.dropout = json_obj['dropout']
        new_layer.recurrent_dropout = json_obj['recurrent_dropout']
        new_layer.activation = activation_lookup[json_obj['activation']]
        new_layer.recurrent_activation = activation_lookup[json_obj['recurrent_activation']]
        new_layer.use_bias = json_obj['use_bias']

        return new_layer

class LSTMBuilder(LayerBuilder):
    def build_layer(self, json_obj, id, layer_type):
        new_layer = LSTM(id, layer_type)

        new_layer.units = json_obj['units']
        new_layer.dropout = json_obj['dropout']
        new_layer.recurrent_dropout = json_obj['recurrent_dropout']
        new_layer.activation = activation_lookup[json_obj['activation']]
        new_layer.recurrent_activation = activation_lookup[json_obj['recurrent_activation']]
        new_layer.implementation = json_obj['implementation']
        new_layer.use_bias = json_obj['use_bias']
        new_layer.go_backwards = json_obj['go_backwards']
        new_layer.stateful = json_obj['stateful']
        new_layer.unroll = json_obj['unroll']

        return new_layer

class ActivationBuilder(LayerBuilder):
    def build_layer(self, json_obj, id, layer_type):
        new_layer = Activation(id, layer_type)

        new_layer.activation = activation_lookup[json_obj['activation']]

        return new_layer

class FlattenBuilder(LayerBuilder):
    def build_layer(self, json_obj, id, layer_type):
        new_layer = Flatten(id, layer_type)

        return new_layer

class DropoutBuilder(LayerBuilder):
    def build_layer(self, json_obj, id, layer_type):
        new_layer = Dropout(id, layer_type)

        return new_layer

class InputBuilder(LayerBuilder):
    def build_layer(self, json_obj, id, layer_type):
        new_layer = Input(id, layer_type)

        return new_layer
