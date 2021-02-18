from nn4mc.datastructures.layer._layer import *
from abc import ABC, abstractmethod

#Parent class for all layer builders
class LayerBuilder(ABC):

    #Builds a layer object from JSON metadata
    @abstractmethod
    def build_layer(self, attributes, id, layer_type):
        pass

################################################################################
#Derived classes

class ConvBuilder(LayerBuilder):
    def build_layer(self, attributes, id, layer_type):

        #Need to figure out which dimension conv
        dims = attributes[0]
        new_layer = Conv1D(id, layer_type)

        # new_layer.filters = 
        # new_layer.kernel_shape =
        # new_layer.strides =
        # new_layer.padding =
        # new_layer.data_format =
        # new_layer.dilation_rate =
        # new_layer.activation =
        # new_layer.use_bias =

        return new_layer

class DenseBuilder(LayerBuilder):
    def build_layer(self, attributes, id, layer_type):
        new_layer = Dense(id, layer_type)

        # new_layer.units =
        # new_layer.activation =
        # new_layer.use_bias =

        return new_layer

class MaxPoolingBuilder(LayerBuilder):
    def build_layer(self, attributes, id, layer_type):
        new_layer = MaxPooling1D(id, layer_type)

        # new_layer.pool_shape =
        # new_layer.strides =
        # new_layer.padding =
        # new_layer.data_format =

        return new_layer

class SimpleRNNBuilder(LayerBuilder):
    def build_layer(self, attributes, id, layer_type):
        new_layer = SimpleRNN(id, layer_type)

        # new_layer.units =
        # new_layer.activation =
        # new_layer.use_bias =
        # new_layer.return_sequences =
        # new_layer.return_state =
        # new_layer.go_backwards =
        # new_layer.stateful =

        return new_layer

class GRUBuilder(LayerBuilder):
    def build_layer(self, attributes, id, layer_type):
        new_layer = GRU(id, layer_type)

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
    def build_layer(self, attributes, id, layer_type):
        new_layer = LSTM(id, layer_type)

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
    def build_layer(self, attributes, id, layer_type):
        new_layer = Activation(id, layer_type)

        # new_layer.activation =

        return new_layer

class DropoutBuilder(LayerBuilder):
    def build_layer(self, attributes, id, layer_type):
        new_layer = Dropout(id, layer_type)

        return new_layer

class FlattenBuilder(LayerBuilder):
    def build_layer(self, attributes, id, layer_type):
        new_layer = Flatten(id, layer_type)

        return new_layer

class InputBuilder(LayerBuilder):
    def build_layer(self, attributes, id, layer_type):
        new_layer = Input(id, layer_type)

        return new_layer
