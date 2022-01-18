#Contains all global file names, and delimiters for use
#in code generator.
activation_lookup = {
    "softmax" :  "0x00",
    "elu" : "0x02",
    "selu" :  "0x03",
    "softplus" : "0x04",
    "softsign" : "0x05",
    "relu" :  "0x06",
    "tanh" : "0x07",
    "sigmoid" :  "0x08",
    "hard_sigmoid" :  "0x09",
    "exponential" :  "0xA",
    "linear" :  "0xB",
    "custom" : "0xC"
}

layer_type_lookup = {
    "dense" : "Dense",
    "gru" : "GRU",
    "conv1d" : "Conv1D",
    "conv2d" : "Conv2D",
    "maxpooling1d" : "MaxPooling1D",
    "maxpooling2d" : "MaxPooling2D",
    "lstm" : "LSTM",
    "simplernn" : "simpleRNN",
    "activation" : "Activation",
    "dropout" : "Dropout",
    "flatten" : "Flatten",
    "input" : "Input",
    "inputlayer" : "InputLayer",
}

class G():

    #Template files
    LAYER_TEMPLATE_HEADER = '/include/layers/'
    LAYER_TEMPLATE_SOURCE = '/src/layers/'

    ACTIVATION_HEADER = '/include/activations.h'
    ACTIVATION_SOURCE = '/src/activations.cpp'

    PARAMETERS_HEADER = '/include/parameters.h'

    NEURAL_NETWORK_HEADER = '/include/nn4mc.h'
    NEURAL_NETWORK_SOURCE = '/src/nn4mc.cpp'

###############################################################################
    #Delimiters

    #Start and end of template
    START_DELIMITER = '<%BEGIN_DEFINITION_TEMPLATE>'
    END_DELIMITER = '<%END_DEFINITION_TEMPLATE>'

    #Start and end of string templates
    START_INIT = '<%BEGIN_INITIALIZE_TEMPLATE>'
    END_INIT = '<%END_INITIALIZE_TEMPLATE>'

    START_FWD = '<%BEGIN_CALL_TEMPLATE>'
    END_FWD = '<%END_CALL_TEMPLATE>'

    #Datatypes
    WEIGHT_DATATYPE_DELIMITER = '<%WEIGHT_DATATYPE_DELIMITER>'
    INDEX_DATATYPE_DELIMITER = '<%INDEX_DATATYPE_DELIMITER>'
    LAYER_DATATYPE_DELIMITER = '<%LAYER_DATATYPE_DELIMITER>'
    ACTIVATION_DATATYPE_DELIMITER = '<%ACTIVATION_DATATYPE_DELIMITER>'

    #Locations for function calls and definitions
    NN_INIT_DELIMITER = '<%BUILD_FUNCTION>'
    NN_FWD_DELIMITER = '<%FWD_FUNCTION>'
    NN_INCLUDE_DELIMITER = '<%INCLUDE>'
    NN_STRUCT_DELIMITER = '<%STRUCTS>'

    W_WEIGHT_DELIMITER = '<%WEIGHT>'

    ACTIVATIONS_DELIMITER = '<%ACTIVATIONS>'

    ACTIVATION_FUNCTION_DELIMITER = '<%ACTIVATION_FUNCTION>'
