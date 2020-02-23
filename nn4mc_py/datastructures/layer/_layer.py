from ._weights import Weight

class Layer:
    #Input and output data shapes: None if not unspecified
    input_shape = [None, None, None]
    output_shape = [None, None, None]

    def __init__(self, id, type='unspecified'):
        self.identifier = id #Unique ID
        self.layer_type = type #Layer type (i.e convolution1D)

        #Think these will probably need to be np arrays
        self.w = Weight()
        self.b = Weight()

    def isInput(self):
        return False

    def generateInit():
        pass

    def generateFwd():
        pass

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other):
        return self.identifier == other.identifier

class Conv1D(Layer):
    filters = 0
    kernel_size = []
    strides = []
    padding = ''
    activation = ''
    use_bias = True

    #dilation_rate = []

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

        return init_string

    #Need to finish this
    def generateFwd(self):
        fwd_string = 'data = fwdConv1D(' + self.identifier + ', data);\n'

        return fwd_string

class Conv2D(Layer):
    filters = 0
    kernel_size = []
    strides = []
    padding = ''
    activation = ''
    use_bias = True

    #dilation_rate = []
    #data_format = ''

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

    def generateFwd(self):
        fwd_string = 'data = fwdConv2D(' + self.identifier + ', data);\n'

        return fwd_string

class Dense(Layer):
    units = 0
    activation = ''
    use_bias = True
    output_size = 0 #NOTE: This is wrong

    #Input shape and output size?
    def generateInit(self):
        init_string = self.identifier + ' = buildDense(&' +\
                    self.w.identifier + '[0], ' +\
                    self.b.identifier + ', ' +\
                    str(self.input_shape[0]) + ', ' +\
                    str(self.output_size) + ', ' +\
                    self.activation + ');\n'

        return init_string

    def generateFwd(self):
        fwd_string = 'data = fwdDense(' + self.identifier + ', data);\n'

        return fwd_string

class Flatten(Layer):
    def generateInit():
        pass

    def generateFwd():
        pass

class MaxPooling1D(Layer):
    pool_size = []
    strides = []
    padding = ''

    #data_format = ''

    def generateInit(self):
        init_string = self.identifier + ' = buildMaxPooling1D(&' +\
                    str(self.pool_size[0]) + ', ' +\
                    str(self.strides[0]) + ', ' +\
                    str(input_shape[0]) + ', ' +\
                    str(input_shape[1]) + ');\n'

        return init_string

    def generateFwd(self):
        fwd_string = 'data = fwdMaxPooling1D(' + self.identifier + ', data);\n'

        return fwd_string

class MaxPooling2D(Layer):
    pool_size = []
    strides = []
    padding = ''

    #data_format = ''

    def generateInit(self):
        init_string = self.identifier + ' = buildMaxPooling2D(&' +\
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

class Dropout(Layer):
    def generateInit():
        pass

    def generateFwd():
        pass

class SimpleRNN(Layer):
    units = 0
    activation = ''
    use_bias = True

    def generateInit():
        pass

    def generateFwd():
        pass

class GRU(Layer):
    units = 0
    dropout = 0.0
    recurrent_dropout = 0.0
    activation = ''
    recurrent_activation = ''
    use_bias = True
    go_backwards = True
    stateful = True
    unrool = True
    reset_after = True

    def generateInit():
        pass

    def generateFwd():
        pass

class LSTM(Layer):
    units = 0
    dropout = 0.0
    implementation = 0
    recurrent_dropout = 0.0
    activation = ''
    recurrent_activation = ''
    use_bias = True
    go_backwards = True
    stateful = True
    unrool = True

    def generateInit():
        pass

    def generateFwd():
        pass

class Input(Layer):
    size = 0

    def isInput(self):
        return True

class Activation(Layer):
    activation = ''
