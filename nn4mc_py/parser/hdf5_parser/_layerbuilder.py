from nn4mc_py.datastructures import Layer
import copy

class LayerBuilder:
    layer_id = ''

    def build_layer(json_obj, id, layer_type):
        pass

class Conv1DBuilder(LayerBuilder):
    def build_layer(json_obj, id, layer_type):
        new_layer = Conv1D(id, layer_type)

        new_layer.filters = json_obj['filters']
        new_layer.kernel_size = copy.copy(json_obj['kernel_size'])
        new_layer.strides = copy.copy(json_obj['strides'])
        new_layer.padding = json_obj['padding']
        new_layer.activation = json_obj['activation']
        new_layer.use_bias = json_obj['use_bias']

        return new_layer

class Conv2DBuilder(LayerBuilder):
    def build_layer(json_obj, id, layer_type):
        new_layer = Conv2D(id, layer_type)

        new_layer.filters = json_obj['filters']
        new_layer.kernel_size = copy.copy(json_obj['kernel_size'])
        new_layer.strides = copy.copy(json_obj['strides'])
        new_layer.padding = json_obj['padding']
        new_layer.activation = json_obj['activation']
        new_layer.use_bias = json_obj['use_bias']

        return new_layer

class DenseBuilder(LayerBuilder):
    def build_layer(json_obj, id, layer_type):
        new_layer = Dense(id, layer_type)

        new_layer.units = json_obj['units']
        new_layer.activation = json_obj['activation']
        new_layer.use_bias = json_obj['use_bias']

        return new_layer

class FlattenBuilder(LayerBuilder):
    def build_layer(json_obj, id, layer_type):
        new_layer = Flatten(id, layer_type)

        return new_layer

class MaxPooling1DBuilder(LayerBuilder):
    def build_layer(json_obj, id, layer_type):
        new_layer = MaxPooling1D(id, layer_type)

        new_layer.pool_size = copy.copy(json_obj['pool_size'])
        new_layer.strides = copy.copy(json_obj['strides'])
        new_layer.padding = json_obj['padding']

        return new_layer

class MaxPooling2DBuilder(LayerBuilder):
    def build_layer(json_obj, id, layer_type):
        new_layer = MaxPooling2D(id, layer_type)

        new_layer.pool_size = copy.copy(json_obj['pool_size'])
        new_layer.strides = copy.copy(json_obj['strides'])
        new_layer.padding = json_obj['padding']

        return new_layer

class DropoutBuilder(LayerBuilder):
    def build_layer(json_obj, id, layer_type):
        new_layer = Dropout(id, layer_type)

        return new_layer

class SimpleRNNBuilder(LayerBuilder):
    def build_layer(json_obj, id, layer_type):
        new_layer = SimpleRNN(id, layer_type)

        new_layer.units
        new_layer.activation
        new_layer.use_bias

        return new_layer

class GRUBuilder(LayerBuilder):
    def build_layer(json_obj, id, layer_type):
        new_layer = GRU(id, layer_type)

        new_layer.units
        new_layer.dropout
        new_layer.recurrent_dropout
        new_layer.activation
        new_layer.recurrent_activation
        new_layer.use_bias
        new_layer.go_backwards
        new_layer.stateful
        new_layer.unrool
        new_layer.reset_after

        return new_layer

class LSTMBuilder(LayerBuilder):
    def build_layer(json_obj, id, layer_type):
        new_layer = LSTM(id, layer_type)

        new_layer.units
        new_layer.dropout
        new_layer.implementation
        new_layer.recurrent_dropout
        new_layer.activation
        new_layer.recurrent_activation
        new_layer.use_bias
        new_layer.go_backwards
        new_layer.stateful
        new_layer.unrool

        return new_layer

#NOTE: These two might need a little more work

class InputBuilder(LayerBuilder):
    def build_layer(json_obj, id, layer_type):
        new_layer = Input(id, layer_type)

        return new_layer

class ActivationBuilder(LayerBuilder):
    def build_layer(json_obj, id, layer_type):
        new_layer = Activation(id, layer_type)

        new_layer.activation = json_obj['activation']

        return new_layer
