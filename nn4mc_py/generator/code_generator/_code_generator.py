from nn4mc_py.datastructures import *
from ._globals import *
import numpy as np
import os

class Generator():
    self.writer_map = {'Conv1D' : 'Conv1DWriter()',
                    'Conv2D' : 'Conv2DWriter()',
                    'Dense' : 'DenseWriter()',
                    'Flatten' : 'FlattenWriter()',
                    'MaxPooling1D' : 'MaxPooling1DWriter()',
                    'MaxPooling2D' : 'MaxPooling2DWriter()',
                    'Dropout' : 'DropoutWriter()',
                    'SimpleRNN' : 'SimpleRNNWriter()',
                    'GRU' : 'GRUWriter()',
                    'LSTM' : 'LSTMWriter()',
                    'Input' : 'InputWriter()',
                    'Activation' : 'ActivationWriter()'}

    TEMPLATE_TYPE = 'c_standard'
    INDEX_DATATYPE = 'int'
    LAYER_OUTPUT_DATATYPE = 'float'
    ACTIVATION_DATATYPE = 'char'
    WEIGHT_DATATYPE = 'const float'

    def __init__(self, nn_obj, output_directory):
        self.nn = nn_obj
        self.output_dir = output_directory

        self.layer_templates = {'header' : [], 'source' : []}
        self.activation_functions = {'header' : [], 'source' : []}

        self.init_templates = {}
        self.fwd_templates = {}

    # Generates the code
    #NOTE:
    def generate(self):
        self.buildFileTree() #Builds output file directory

        self.processTemplates() #Processes required templates

        self.processLayers() #Processes layers

        self.dump() #Dumps output code

    # Iterates through graph to extract which layers and
    # activations are required. Also replaces any delimiters
    # that can be at this point
    #NOTE:
    def processTemplates(self):
        layers = []
        activations = []

        #Get lists of all activations and layers
        for node in self.neural_network.iterate():
            type = node.layer.layer_type
            activation = node.layer.activation

            if type not in layers:
                layers.append(type)
            if activation != '' and activation not in activations:
                activations.append(activation)

        #For each type scrape file and replace delimiters
        for layer_type in layers:
            #Replace delimiters and add to layer_templates
            with open(LAYER_TEMPLATE_HEADER + layer_type, 'r') as header:
                contents = header.read()

                contents = self.replaceDelimiters(contents)

                self.layer_templates['header'].append(contents)

            #Replace delimiters, and extract call and fwd templates
            with open(LAYER_TEMPLATE_SOURCE + layer_type, 'r') as source:
                contents = source.read()

                init, fwd = self.getFunctionStrings(contents)
                contents = self.replaceDelimiters(contents)

                self.init_templates[layer_type] = init
                self.fwd_templates[layer_type] = fwd
                self.layer_templates['source'].append(contents)

        #For each type scrape and replace delimiters
        for act_type in activations:
            with open(ACTIVATION_HEADER, 'r') as header:
                contents = header.read()

                contents = self.replaceDelimiters(contents)

                self.layer_templates['header'].append(contents)

            with open(ACTIVATION_SOURCE, 'r') as source:
                contents = source.read()

                contents = self.replaceDelimiters(contents)

                self.layer_templates['source'].append(contents)

    # Iterates through graph to extract all metadata and
    # weight data and place in appropriate templates to
    # be dumped.
    #NOTE:
    def processLayers(self):
        #For each node we need to deal with the weights and biases
        #and write the init and fwd functions
        for node in self.neural_network.iterate():
            #Deal with weights and bias stuff

            #Deal with writing the layer
            pass

    # Builds the output file structure
    #NOTE: Need to add more error handling
    def buildFileTree(self):
        directories = []
        directories.append(self.output_dir + '/nn4mc')
        directories.append(self.output_dir + '/nn4mc/include')
        directories.append(self.output_dir + '/nn4mc/include/layers')
        directories.append(self.output_dir + '/nn4mc/source')
        directories.append(self.output_dir + '/nn4mc/include/layers')

        try:
            for dir in directories:
                os.mkdir(dir)
        except:
            print('Some error occured.')

    # Dumps all files into output structure
    #NOTE:
    def dump(self):
        pass

    def replaceDelimiters(self, contents):
        start = contents.find(self.START_DELIMITER)
        end = contents.find(self.END_DELIMITER)

        start += self.START_DELIMITER.len()

        contents = contents[start:end]

        contents.replace(self.WEIGHT_DATATYPE_DELIMITER, self.WEIGHT_DATATYPE)
        contents.replace(self.INDEX_DATATYPE_DELIMITER, self.INDEX_DATATYPE)
        #I think there might be more to this than I am thinking
        contents.replace(self.LAYER_DATATYPE_DELIMITER, self.LAYER_DATATYPE)
        contents.replace(self.ACTIVATION_DATATYPE_DELIMITER, self.ACTIVATION_DATATYPE)

        return contents

    def getFunctionStrings(self, contents):
        start = contents.find(self.START_INIT_DELIMITER)
        end = contents.find(self.END_INIT_DELIMITER)

        start += self.START_INIT_DELIMITER.len()

        init = contents[start:end]

        start = contents.find(self.START_CALL_DELIMITER)
        end = contents.find(self.END_CALL_DELIMITER)

        start += self.END_CALL_DELIMITER.len()

        fwd = contents[start:end]

        return init, fwd
