#Contains all global file names, and delimiters for use
#in code generator.
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
