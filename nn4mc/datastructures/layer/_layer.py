from ._weights import Weight
from abc import ABC, abstractmethod

#Parent class for all available layer types
class Layer(ABC):
    #Input and output data shapes: None if not unspecified
    input_shape = None # we might have to change this
    output_shape = None # we might have to change this
    w = None
    b = None

    def __init__(self, id, type='unspecified'):
        self.identifier = id #Unique ID
        self.layer_type = type #Layer type (i.e conv1d)

    #Add weight and bias parameters
    #Takes tuple of (id, values) for weights and biases
    def addParameters(self, type, data):
        if type == 'weight':
            self.w = Weight(data[0], data[1])

        elif type == 'bias':
            self.b = Weight(data[0], data[1])

        elif type == 'weight_rec':
            self.w_rec = Weight(data[0], data[1])

    def isInput(self): #Defualt behavior is not input
        return False

    @abstractmethod
    def computeOutShape(self):
        pass

    @abstractmethod
    def generateInit(): #For derived classes
        pass

    @abstractmethod
    def generateFwd(): #For derived classes
        pass

    def __hash__(self): #Hashes on the identifier
        return hash(self.identifier)

    def __eq__(self, other): #Equality on the unique identifier
        return self.identifier == other.identifier

################################################################################
#Derived classes (i.e specific layer types)

class Conv1D(Layer):
    filters = 0
    kernel_size = []
    strides = []
    dilation_rate = []
    padding = ''
    data_format = ''
    activation = ''
    use_bias = True

    def generateInit(self):
        init_string = self.identifier + ' = buildConv1D(&' +\
                    self.w.identifier + '[0], ' +\
                    self.b.identifier + ', ' +\
                    str(self.kernel_size[0]) + ', ' +\
                    str(self.strides[0]) + ', ' +\
                    str(self.input_shape[0]) + ', ' +\
                    str(self.input_shape[1]) + ', ' +\
                    str(self.filters) + ', ' +\
                    self.activation + ');\n'
                    #Need to add padding, data_format, and dilation_rate

        return init_string

    def generateFwd(self):
        fwd_string = 'data = fwdConv1D(' + self.identifier + ', data);\n'

        return fwd_string

    # (int)(layer.input_shape[0] - layer.kernel_shape[0] + 1)
    def computeOutShape(self, input_shape = None):
        self.input_shape = input_shape
        output_shape = [0.0]*2
        if input_shape is not None:
            output_shape[0] = input_shape[0] - self.kernel_size[0] + 1
            output_shape[1] = self.filters
        self.output_shape = output_shape

        return output_shape

class Conv2D(Layer):
    filters = 0
    kernel_size = []
    strides = []
    dilation_rate = []
    padding = ''
    data_format = ''
    activation = ''
    use_bias = True

    def generateInit(self):
        init_string = self.identifier + ' = buildConv2D(&' +\
                    self.w.identifier + '[0], ' +\
                    self.b.identifier + ', ' +\
                    str(self.kernel_size[0]) + ', ' +\
                    str(self.kernel_size[1]) + ', ' +\
                    str(self.filters) + ', ' +\
                    str(self.strides[0]) + ', ' +\
                    str(self.strides[1]) + ', ' +\
                    str(self.input_shape[0]) + ', ' +\
                    str(self.input_shape[1]) + ', ' +\
                    str(self.input_shape[2]) + ', ' +\
                    self.activation + ');\n'
                    #Need to add padding, data_format, and dilation_rate

    def generateFwd(self):
        fwd_string = 'data = fwdConv2D(' + self.identifier + ', data);\n'

        return fwd_string

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        output_shape = [0.0]*3
        if input_shape is not None:
            output_shape[0] = input_shape[0] - self.kernel_size[0] + 1
            output_shape[1] = input_shape[1] - self.kernel_size[1] + 1
            output_shape[2] = self.filters
        self.output_shape = output_shape

        return output_shape

class Dense(Layer):
    units = 0
    activation = ''
    use_bias = True

    def generateInit(self):
        init_string = self.identifier + ' = buildDense(&' +\
                    self.w.identifier + '[0], ' +\
                    self.b.identifier + ', ' +\
                    str(self.input_shape[0]) + ', ' +\
                    str(self.output_shape[0]) + ', ' +\
                    self.activation + ');\n'

        return init_string

    def generateFwd(self):
        fwd_string = 'data = fwdDense(' + self.identifier + ', data);\n'

        return fwd_string

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = [self.b.values.shape[0]]

        return self.output_shape

class MaxPooling1D(Layer):
    pool_size = []
    strides = []
    padding = ''
    data_format = ''

    def generateInit(self):
        init_string = self.identifier + ' = buildMaxPooling1D(' +\
                    str(self.pool_size[0]) + ', ' +\
                    str(self.strides[0]) + ', ' +\
                    str(input_shape[0]) + ', ' +\
                    str(input_shape[1]) + ');\n'

        return init_string

    def generateFwd(self):
        fwd_string = 'data = fwdMaxPooling1D(' + self.identifier + ', data);\n'

        return fwd_string

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape

        return self.output_shape

class MaxPooling2D(Layer):
    pool_size = []
    strides = []
    padding = ''
    data_format = ''

    def generateInit(self):
        init_string = self.identifier + ' = buildMaxPooling2D(' +\
                    str(self.pool_size[0]) + ', ' +\
                    str(self.pool_size[1]) + ', ' +\
                    str(self.strides[0]) + ', ' +\
                    str(input_shape[0]) + ', ' +\
                    str(input_shape[1]) + ', ' +\
                    str(input_shape[2]) + ');\n'

        return init_string

    def generateFwd(self):
        fwd_string = 'data = fwdMaxPooling2D(' + self.identifier + ', data);\n'

        return fwd_string

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape= input_shape

        return input_shape

################################################################################
#TODO: Check on all fwd and init generators, they arent finished in nn4mc std

class SimpleRNN(Layer):
    units = 0
    activation = ''
    use_bias = True
    return_sequences = False
    return_state = False
    go_backwards = False
    stateful = False

    def generateInit():
        init_string = self.identifier + ' = buildSimpleRNN(&' +\
                    self.w.identifier + '[0], ' +\
                    self.w_rec.identifier + '[0], ' +\
                    self.b.identifier + ',' +\
                    str(self.input_shape[0]) + ',' +\
                    str(self.input_shape[1]) + ',' +\
                    str(self.output_shape[0]) + ',' +\
                    self.activation + ',' +\
                    self.go_backwards + ');\n'

        return init_string

    def generateFwd():
        fwd_string = 'data = fwdSimpleRNN(' + self.identifier + ', data);\n'

        return fwd_string

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = [self.b.values.shape[0]]

        return self.output_shape

class GRU(Layer):
    units = 0
    dropout = 0.0
    recurrent_dropout = 0.0
    activation = ''
    recurrent_activation = ''
    use_bias = True
    go_backwards = False
    stateful = False
    unroll = False
    reset_after = False

    def generateInit():
        init_string = self.identifier + ' = buildGRU(&' +\
                    self.w.identifier + '[0], ' +\
                    self.w_rec.identifier + '[0], ' +\
                    self.b.identifier + ',' +\
                    str(self.input_shape[0]) + ',' +\
                    str(self.input_shape[1]) + ',' +\
                    str(self.output_shape[0]) + ',' +\
                    self.activation + ',' +\
                    self.recurrent_activation + ',' +\
                    str(self.dropout) + ',' +\
                    str(self.recurrent_dropout) + ',' +\
                    self.go_backwards + + ');\n'

        return init_string

    def generateFwd():
        fwd_string = 'data = fwdGRU(' + self.identifier + ', data);\n'

        return fwd_string

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = [self.b.values.shape[0]]

        return self.output_shape

class LSTM(Layer):
    units = 0
    dropout = 0.0
    implementation = 0
    recurrent_dropout = 0.0
    activation = ''
    recurrent_activation = ''
    use_bias = True
    go_backwards = False
    stateful = False
    unroll = False

    def generateInit():
        init_string = self.identifier + ' = buildGRU(&' +\
                    self.w.identifier + '[0], ' +\
                    self.w_rec.identifier + '[0], ' +\
                    self.b.identifier + ',' +\
                    str(self.input_shape[0]) + ',' +\
                    str(self.input_shape[1]) + ',' +\
                    str(self.output_shape[0]) + ',' +\
                    self.activation + ',' +\
                    self.recurrent_activation + ',' +\
                    str(self.dropout) + ',' +\
                    str(self.recurrent_dropout) + ',' +\
                    self.go_backwards + + ');\n'

        return init_string

    def generateFwd():
        fwd_string = 'data = fwdLSTM(' + self.identifier + ', data);\n'

        return fwd_string

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = [self.b.values.shape[0]]

        return self.output_shape

################################################################################
# NOTE: Not sure about these
# I think some stuff needs to be changed, at least in the templates, but they
# are implemented just uncertain about some stuff.
class Flatten(Layer):
    def generateInit():
        return ''

    def generateFwd():
        return ''

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        temp = 1.0
        for i in range(len(input_shape)):
            temp *= input_shape[i]
        self.output_shape = temp
        return temp

class Activation(Layer):
    activation = ''

    def generateInit():
        pass

    def generateFwd():
        pass

    def computeOutShape(self, input_shape):
        pass

class Dropout(Layer):
    def generateInit():
        return ''

    def generateFwd():
        return ''

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        return self.output_shape

class Input(Layer):
    size = 0

    def generateInit(self):
        return ''

    def generateFwd(self):
        return ''

    def computeOutShape(self, input_shape):
        return input_shape

    def isInput(self):
        return True
