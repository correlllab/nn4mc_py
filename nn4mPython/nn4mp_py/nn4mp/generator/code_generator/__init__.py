class Weight_Generator:
    pass

class Network_Generator:
    pass

class Layer_Generator:
    def __init__(self, layer_include, layer_source, weight_include,
    network_include, network_source, param_datatype, index_datatype):
        self.template_include_dir = layer_include
        self.template_source_dir = layer_source
        self.weight_datatype = param_datatype
        self.index_datatype = index_datatype

        self.WEIGHT_DATATYPE_DELIMITER = '<%WEIGHT_DATATYPE_DELIMITER>'
        self.INDEX_DATATYPE_DELIMITER = '<%INDEX_DATATYPE_DELIMITER>'
        self.START_DELIMITER = '<%BEGIN_DEFINITION_TEMPLATE'
        self.END_DELIMITER = '<%END_DEFINITION_TEMPLATE>'
        self.INIT_START_DELIMITER = '<%BEGIN_INIT_TEMPLATE>'
        self.INIT_END_DELIMITER = '<%END_INIT_TEMPLATE>'
        self.FWD_START_DELIMITER = '<%BEGIN_FWD_TEMPLATE>'
        self.FWD_END_DELIMITER = '<%END_FWD_TEMPLATE>'

        self.init_strings = {}
        self.fwd_strings = {}
        self.include_strings = {}

        self.include_files = {}
        self.source_files = {}

        weight_gen = Weight_Generator(weight_include)
        network_gen = Network_Generator(network_include, network_source)

class Code_Generator:
    def __init__(self, nn, template_dir, ouput_dir):
        self.template_dir = template_dir
        self.output_dir = output_dir

        self.neural_network = nn

        self.LAYER_TEMPLATE_INCLUDE_DIR = '/include/layers'
        self.LAYER_TEMPLATE_SRC_DIR = '/src/layers'
        self.INCLUDE = '/include'
        self.SOURCE = '/src'
        self.PARAMETER_FILE = '/neural_network_params.h'
        self.HEADER_FILE = '/neural_network.h'
        self.SOURCE_FILE = '/neural_network.c'
        self.PARAMETER_DATATYPE = 'const float'
        self.LAYER_DATATYPE = 'float'
        self.INDEX_DATATYPE = 'int'

        layer_include = template_dir + self.LAYER_TEMPLATE_INCLUDE_DIR
        layer_source = template_dir + self.LAYER_TEMPLATE_SRC_DIR

        weight_include = template_dir + self.INCLUDE + self.PARAMETER_FILE

        network_include = template_dir + self.INCLUDE + self.HEADER_FILE
        network_source = template_dir + self.SOURCE + self.SOURCE_FILE

        layer_gen = Layer_Generator(layer_include, layer_source,
        weight_include, network_include, network_source,
        self.PARAMETER_DATATYPE, self.INDEX_DATATYPE)

    #Imported functions
    from ._code_generator import generate
    from ._code_generator import dump
