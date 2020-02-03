#Constants

#Template files
LAYER_TEMPLATE_HEADER = 'include/layers'
LAYER_TEMPLATE_SOURCE = 'src/layers'

ACTIVATION_HEADER = 'include/activation_func.h'
ACTIVATION_SOURCE = 'src/activation_func.cpp'

PARAMETERS_HEADER = 'include/parameters.h'

NEURAL_NETWORK_HEADER = 'neural_network.h'
NEURAL_NETWORK_SOURCE = 'neural_network.cpp'

#Delimiters
#Start and end of template
START_DELIMITER = '<%BEGIN_DEFINITION_TEMPLATE>'
END_DELIMITER = '<%END_DEFINITION_TEMPLATE>'

#Datatypes
WEIGHT_DATATYPE_DELIMITER = '<%WEIGHT_DATATYPE_DELIMITER>'
INDEX_DATATYPE_DELIMITER = '<%INDEX_DATATYPE_DELIMITER>'
LAYER_DATATYPE_DELIMITER = '<%LAYER_DATATYPE_DELIMITER>'
ACTIVATION_DATATYPE_DELIMITER = '<%ACTIVATION_DATATYPE_DELIMITER>'

#Weight stuff
WEIGHT_NAME_DELIMITER = '<%WEIGHT_NAME_DELIMITER>'
WEIGHT_INDEX_DELIMITER = '<%WEIGHT_INDEX_DELIMITER>'
WEIGHT_DATA_DELIMITER = '<%WEIGHT_DATA_DELIMITER>'

#Template function call
START_CALL_DELIMITER = '<%START_CALL_TEMPLATE>'
END_CALL_DELIMITER = '<%END_CALL_TEMPLATE>'

#Template initialize call
START_INIT_DELIMITER = '<%START_INITIALIZE_TEMPLATE'
END_INIT_DELIMITER = '<%END_INITIALIZE_TEMPLATE>'

#Locations for function calls and definitions
NN_INIT_DELIMITER = '<%BUILD_FUNCTION>'
NN_FWD_DELIMITER = '<%FWD_FUNCTION>'
NN_INCLUDE_DELIMITER = '<%INCLUD>'
NN_STRUC_DELIMITER = '<%STRUCTS>'
