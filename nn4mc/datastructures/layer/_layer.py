from ._weights import Weight
from abc import ABC, abstractmethod
from ._globals import G

#Parent class for all available layer types
class Layer(ABC):
    #Input and output data shapes: None if not unspecified
    input_shape = None
    output_shape = None
    params = {'w':Weight(None, None), 'b':Weight(None, None), 'w_rec':Weight(None, None)}

    def __init__(self, id):
        self.identifier = id #Unique ID

    #Add weight and bias parameters
    #Takes tuple of (id, values) for weights and biases
    def addParameters(self, type, data):
        if type == 'weight':
            self.params['w'] = Weight(data[0], data[1])

        elif type == 'bias':
            self.params['b'] = Weight(data[0], data[1])

        elif type == 'weight_rec':
            self.params['w_rec'] = Weight(data[0], data[1])

    def getParameters(self):
        param_string = ''
        for weight in self.params.values():
            if weight.identifier!=None:
                param_string = param_string + weight.getParams()

        return param_string

    def generateAct(self):
        if self.activation!='' and self.activation!='linear':
            if self.activation=='softmax':
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
            val = eval(G.delim_map[meta])

            temp_string = temp_string.replace(temp_string[start:end+len(G.end_delim)],
            val)

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
    layer_type = conv1d

    filters = 0
    kernel_shape = []
    strides = []
    dilation_rate = []
    padding = ''
    data_format = ''
    activation = ''
    use_bias = True

    # def generateInit(self):
    #     init_string = self.identifier + ' = buildConv1D(&' +\
    #                 self.param['w'].identifier + '[0], ' +\
    #                 self.params['b'].identifier + ', ' +\
    #                 str(self.kernel_shape[0]) + ', ' +\
    #                 str(self.strides[0]) + ', ' +\
    #                 str(self.input_shape[0]) + ', ' +\
    #                 str(self.input_shape[1]) + ', ' +\
    #                 str(self.filters) + ', ' +\
    #                 self.activation + ');\n'
    #
    #     return init_string
    #
    # def generateFwd(self):
    #     fwd_string = 'data = fwdConv1D(' + self.identifier + ', data);\n'
    #
    #     return fwd_string

    def computeOutShape(self, input_shape = None):
        self.input_shape = input_shape
        output_shape = [0.0]*2
        if input_shape is not None:
            output_shape[0] = input_shape[0] - self.kernel_shape[0] + 1
            output_shape[1] = self.filters
        self.output_shape = output_shape

        return output_shape

class Conv2D(Layer):
    layer_type = conv2d

    filters = 0
    kernel_shape = []
    strides = []
    dilation_rate = []
    padding = ''
    data_format = ''
    activation = ''
    use_bias = True

    # def generateInit(self):
    #     init_string = self.identifier + ' = buildConv2D(&' +\
    #                 self.params['w'].identifier + '[0], ' +\
    #                 self.params['b'].identifier + ', ' +\
    #                 str(self.kernel_shape[0]) + ', ' +\
    #                 str(self.kernel_shape[1]) + ', ' +\
    #                 str(self.filters) + ', ' +\
    #                 str(self.strides[0]) + ', ' +\
    #                 str(self.strides[1]) + ', ' +\
    #                 str(self.input_shape[0]) + ', ' +\
    #                 str(self.input_shape[1]) + ', ' +\
    #                 str(self.input_shape[2]) + ', ' +\
    #                 self.activation + ');\n'
    #
    # def generateFwd(self):
    #     fwd_string = 'data = fwdConv2D(' + self.identifier + ', data);\n'
    #
    #     return fwd_string

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
    layer_type = dense

    units = 0
    activation = ''
    use_bias = True

    # def generateInit(self):
    #     init_string = self.identifier + ' = buildDense(&' +\
    #                 self.params['w'].identifier + '[0], ' +\
    #                 self.params['b'].identifier + ', ' +\
    #                 str(self.input_shape[0]) + ', ' +\
    #                 str(self.output_shape[0]) + ', ' +\
    #                 self.activation + ');\n'
    #
    #     return init_string
    #
    # def generateFwd(self):
    #     fwd_string = 'data = fwdDense(' + self.identifier + ', data);\n'
    #
    #     return fwd_string

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = [self.params['b'].values.shape[0]]

        return self.output_shape

class MaxPooling1D(Layer):
    layer_type = maxpooling1d

    pool_shape = []
    strides = []
    padding = ''
    data_format = ''

    # def generateInit(self):
    #     init_string = self.identifier + ' = buildMaxPooling1D(' +\
    #                 str(self.pool_shape[0]) + ', ' +\
    #                 str(self.strides[0]) + ', ' +\
    #                 str(input_shape[0]) + ', ' +\
    #                 str(input_shape[1]) + ');\n'
    #
    #     return init_string
    #
    # def generateFwd(self):
    #     fwd_string = 'data = fwdMaxPooling1D(' + self.identifier + ', data);\n'
    #
    #     return fwd_string

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape

        return self.output_shape

class MaxPooling2D(Layer):
    layer_type = maxpooling2d

    pool_shape = []
    strides = []
    padding = ''
    data_format = ''

    # def generateInit(self):
    #     init_string = self.identifier + ' = buildMaxPooling2D(' +\
    #                 str(self.pool_shape[0]) + ', ' +\
    #                 str(self.pool_shape[1]) + ', ' +\
    #                 str(self.strides[0]) + ', ' +\
    #                 str(input_shape[0]) + ', ' +\
    #                 str(input_shape[1]) + ', ' +\
    #                 str(input_shape[2]) + ');\n'
    #
    #     return init_string
    #
    # def generateFwd(self):
    #     fwd_string = 'data = fwdMaxPooling2D(' + self.identifier + ', data);\n'
    #
    #     return fwd_string

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape= input_shape

        return input_shape

################################################################################
#TODO: Check on all fwd and init generators, they arent finished in nn4mc std
#TODO: Check on all parameters

class SimpleRNN(Layer):
    layer_type = simplernn

    units = 0
    activation = ''
    use_bias = True
    return_sequences = False
    return_state = False
    go_backwards = False
    stateful = False

    # def generateInit():
    #     init_string = self.identifier + ' = buildSimpleRNN(&' +\
    #                 self.params['w'].identifier + '[0], ' +\
    #                 self.params['w_rec'].identifier + '[0], ' +\
    #                 self.params['b'].identifier + ',' +\
    #                 str(self.input_shape[0]) + ',' +\
    #                 str(self.input_shape[1]) + ',' +\
    #                 str(self.output_shape[0]) + ',' +\
    #                 self.activation + ',' +\
    #                 self.go_backwards + ');\n'
    #
    #     return init_string
    #
    # def generateFwd():
    #     fwd_string = 'data = fwdSimpleRNN(' + self.identifier + ', data);\n'
    #
    #     return fwd_string

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = [self.params['b'].values.shape[0]]

        return self.output_shape

class GRU(Layer):
    layer_type = gru

    units = 0
    activation = ''
    recurrent_activation = ''
    use_bias = True
    go_backwards = False
    stateful = False
    unroll = False
    reset_after = False

    # def generateInit():
    #     init_string = self.identifier + ' = buildGRU(&' +\
    #                 self.params['w'].identifier + '[0], ' +\
    #                 self.params['w_rec'].identifier + '[0], ' +\
    #                 self.params['b'].identifier + ',' +\
    #                 str(self.input_shape[0]) + ',' +\
    #                 str(self.input_shape[1]) + ',' +\
    #                 str(self.output_shape[0]) + ',' +\
    #                 self.activation + ',' +\
    #                 self.recurrent_activation + ',' +\
    #                 str(self.dropout) + ',' +\
    #                 str(self.recurrent_dropout) + ',' +\
    #                 self.go_backwards + + ');\n'
    #
    #     return init_string
    #
    # def generateFwd():
    #     fwd_string = 'data = fwdGRU(' + self.identifier + ', data);\n'
    #
    #     return fwd_string

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = [self.params['b'].values.shape[0]]

        return self.output_shape

class LSTM(Layer):
    layer_type = lstm

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

    # def generateInit():
    #     init_string = self.identifier + ' = buildGRU(&' +\
    #                 self.params['w'].identifier + '[0], ' +\
    #                 self.params['w_rec'].identifier + '[0], ' +\
    #                 self.params['b'].identifier + ',' +\
    #                 str(self.input_shape[0]) + ',' +\
    #                 str(self.input_shape[1]) + ',' +\
    #                 str(self.output_shape[0]) + ',' +\
    #                 self.activation + ',' +\
    #                 self.recurrent_activation + ',' +\
    #                 str(self.dropout) + ',' +\
    #                 str(self.recurrent_dropout) + ',' +\
    #                 self.go_backwards + + ');\n'
    #
    #     return init_string
    #
    # def generateFwd():
    #     fwd_string = 'data = fwdLSTM(' + self.identifier + ', data);\n'
    #
    #     return fwd_string

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = [self.params['b'].values.shape[0]]

        return self.output_shape

################################################################################
#TODO: Check the implementation of this in code generator
class Activation(Layer):
    layer_type = activation

    activation = ''

    # def generateInit():
    #     pass
    #
    # def generateFwd():
    #     pass

    def computeOutShape(self, input_shape):
        pass

#NOTE: No template or anything for this as everything is already flattened
class Flatten(Layer):
    layer_type = flatten

    # def generateInit():
    #     return ''
    #
    # def generateFwd():
    #     return ''

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        temp = 1.0
        for i in range(len(input_shape)):
            temp *= input_shape[i]
        self.output_shape = temp
        return temp

class Dropout(Layer):
    layer_type = dropout

    # def generateInit():
    #     return ''
    #
    # def generateFwd():
    #     return ''

    def computeOutShape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        return self.output_shape

#NOTE: This is really just useful in the graph to find starting point
class Input(Layer):
    layer_type = input

    size = 0

    # def generateInit(self):
    #     return ''
    #
    # def generateFwd(self):
    #     return ''

    def computeOutShape(self, input_shape):
        return input_shape

    def isInput(self):
        return True
