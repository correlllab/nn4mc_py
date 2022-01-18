from ._weights import Weight
from abc import ABC, abstractmethod
from ._globals import G

#Parent class for all available layer types
class Layer(ABC):
    #Input and output data shapes: None if not unspecified
    def __init__(self, id, type=None):
        self.identifier = id #Unique ID
        self.layer_type = type #Layer type (i.e conv1d)
        self.input_shape = None
        self.output_shape = None
        self.params = {
            'w': Weight(None, None),
            'b': Weight(None, None),
            'w_rec': Weight(None, None)
        }

    # add weight and bias parameters
    # takes tuple of (id, values) for weights and biases
    def setParameters(self, type, data):
        if type == 'weight':
            self.params['w'] = Weight(data[0], data[1])

        elif type == 'bias':
            self.params['b'] = Weight(data[0], data[1])

        elif type == 'weight_rec':
            self.params['w_rec'] = Weight(data[0], data[1])

    def getParameters(self):
        param_string = ''
        for weight in self.params.values():
            param_string = param_string + weight.getParams()
        return param_string

    def generateAct(self):
        if self.activation!='' and self.activation!='linear':
            if self.activation == 'softmax':
                act_string = 'data = ' + self.activation +\
                 '(data, ' + self.output_shape + ' );\n'
            else:
                act_string = 'data = ' + self.activation + '(data);\n'

            return act_string
        else:
            return ''

    def generateCall(self, temp_string): #For derived classes
        start = temp_string.find(G.start_delim)
        end = temp_string.find(G.end_delim)

        while(start != -1):
            meta = temp_string[start+len(G.start_delim):end]
            if meta in G.delim_map.keys():
                try:
                    val = eval(G.delim_map[meta])
                except:
                    print(temp_string)
                    quit()
                if val is not None:
                    temp_string = temp_string.replace(temp_string[start:end+len(G.end_delim)],
                                                    val)
                else:
                    break
            start = temp_string.find(G.start_delim)
            end = temp_string.find(G.end_delim)

        return temp_string

    def isInput(self): #Defualt behavior is not input
        return False

    @abstractmethod
    def computeOutShape(self):
        pass

    def __hash__(self): #Hashes on the identifier
        return hash(self.identifier)

    def __eq__(self, other): #Equality on the unique identifier
        return self.identifier == other.identifier

################################################################################
#Derived classes (i.e specific layer types)

class Conv1D(Layer):
    def __init__(self, id, layer_type):
        self.filters = 0
        self.kernel_shape = []
        self.strides = []
        self.dilation_rate = []
        self.padding = ''
        self.data_format = ''
        self.activation = ''
        self.use_bias = True
        super().__init__(id, layer_type)

    def computeOutShape(self, input_shape = None):
        self.input_shape = input_shape
        output_shape = [0.0]*2
        if input_shape is not None:
            output_shape[0] = input_shape[0] - self.kernel_shape[0] + 1
            output_shape[1] = self.filters
        self.output_shape = output_shape

        return output_shape

class Conv2D(Layer):
    def __init__(self, id, layer_type):
        self.filters = 0
        self.kernel_shape = []
        self.strides = []
        self.dilation_rate = []
        self.padding = ''
        self.data_format = ''
        self.activation = ''
        self.use_bias = True
        super().__init__(id, layer_type)

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        output_shape = [0.0]*3
        if input_shape is not None:
            output_shape[0] = input_shape[0] - self.kernel_shape[0] + 1
            output_shape[1] = input_shape[1] - self.kernel_shape[1] + 1
            output_shape[2] = self.filters
        self.output_shape = output_shape

        return output_shape

class Dense(Layer):
    def __init__(self, id, layer_type):
        self.units = 0
        self.activation = ''
        self.use_bias = True
        super().__init__(id, layer_type)

    def computeOutShape(self, input_shape):
        self.input_shape = [input_shape]
        self.output_shape = [self.units]
        return self.output_shape

class MaxPooling1D(Layer):
    def __init__(self, id, layer_type):
        self.pool_shape = []
        self.strides = []
        self.padding = ''
        self.data_format = ''
        super().__init__(id, layer_type)

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        return self.output_shape

class MaxPooling2D(Layer):
    def __init__(self, id, layer_type):
        self.pool_shape = []
        self.strides = []
        self.padding = ''
        self.data_format = ''
        super().__init__(id, layer_type)

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape= input_shape

        return input_shape

################################################################################
#TODO(RS-Coop): Check on all fwd and init generators, they arent finished in nn4mc std
#TODO(RS-Coop): Check on all parameters

class SimpleRNN(Layer):
    def __init(self, id, layer_type):
        self.units = 0
        self.activation = ''
        self.use_bias = True
        self.return_sequences = False
        self.return_state = False
        self.go_backwards = False
        self.stateful = False
        super().__init__(id, layer_type)

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = [self.params['b'].values.shape[0] // 3]

        return self.output_shape

class GRU(Layer):
    def __init__(self, id, layer_type):
        self.units = 0
        self.activation = ''
        self.recurrent_activation = ''
        self.use_bias = True
        super().__init__(id, layer_type)

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = [self.units]
        return self.output_shape


class LSTM(Layer):
    def __init__(self, id, layer_type):
        self.units = 0
        self.dropout = 0.0
        self.implementation = 0
        self.recurrent_dropout = 0.0
        self.activation = ''
        self.recurrent_activation = ''
        self.use_bias = True
        self.go_backwards = False
        self.stateful = False
        self.unroll = False
        super().__init__(id, layer_type)

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = [self.units]

        return self.output_shape

################################################################################
#TODO: Check the implementation of this in code generator
class Activation(Layer):
    def __init__(self, id, layer_type):
        self.activation = ''
        super().__init__(id, layer_type)

    def computeOutShape(self, input_shape):
        pass

#NOTE: No template or anything for this as everything is already flattened
class Flatten(Layer):
    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        temp = 1.0
        for i in range(len(input_shape)):
            temp *= input_shape[i]
        self.output_shape = temp
        return temp

class Dropout(Layer):
    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        return self.output_shape

#NOTE: This is really just useful in the graph to find starting point
class Input(Layer):
    def __init__(self, id, layer_type):
        self.size = 0
        super().__init__(id, layer_type)

    def computeOutShape(self, input_shape):
        return input_shape

    def isInput(self):
        return True