from nn4mc_py.datastructures import *
from ._globals import *
import numpy as np
import os

class Generator():
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
        self.parameters_template = ''
        self.neural_network_template = {'header' : '', 'source' : ''}

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
            with open(LAYER_TEMPLATE_HEADER + layer_type + '.template', 'r') as header:
                contents = header.read()
                contents = self.replaceDelimiters(contents)

                self.layer_templates['header'].append(contents)

            #Replace delimiters, and extract call and fwd templates
            with open(LAYER_TEMPLATE_SOURCE + layer_type + '.template', 'r') as source:
                contents = source.read()
                contents = self.replaceDelimiters(contents)

                self.layer_templates['source'].append(contents)

        #For each type scrape and replace delimiters
        for act_type in activations:
            with open(ACTIVATION_HEADER + '.template', 'r') as header:
                contents = header.read()
                contents = self.replaceDelimiters(contents)

                self.layer_templates['header'].append(contents)

            with open(ACTIVATION_SOURCE + '.template', 'r') as source:
                contents = source.read()
                contents = self.replaceDelimiters(contents)

                self.layer_templates['source'].append(contents)

        #Scrape weight file
        with open(PARAMETERS_HEADER + '.template', 'r') as params:
            contents = params.read()
            contents = self.replaceDelimiters(contents)

            self.parameters_template = contents

        #Scrape nn4mc main file and add include statements
        with open(NEURAL_NETWORK_HEADER + '.template', 'r') as header:
            contents = header.read()
            contents = self.replaceDelimiters(contents)

            include_string = ''
            #Might need to be edited
            for layer_type in layers:
                include_string = include_string + '#include ' +\
                                layer_type + '.h\n'

            contents = contents.replace(NN_INCLUDE_DELIMITER, include_string)

            self.neural_network_template['header'] = contents

        with open(NEURAL_NETWORK_SOURCE + '.template', 'r') as source:
            contents = source.read()
            contents = self.replaceDelimiters(contents)

            self.neural_network_template['source'] = contents

    # Iterates through graph to extract all metadata and
    # weight data and place in appropriate templates to
    # be dumped.
    #NOTE:
    def processLayers(self):
        #For each node we need to deal with the weights and biases
        #and write the init and fwd functions
        for node in self.neural_network.iterate():
            weight_string = node.layer.w.getParams()
            bias_string = node.layer.b.getParams()
            init_string = node.layer.generateInit()
            fwd_string = node.layer.generateFwd()

            #Deal with weights and bias stuff
            self.parameters_template = self.parameters_template.replace(
            W_WEIGHT_DELIMITER, weight_string + W_WEIGHT_DELIMITER)
            self.parameters_template = self.parameters_template.replace(
            W_WEIGHT_DELIMITER, bias_string + W_WEIGHT_DELIMITER)

            #Deal with writing the layer


        #Remove the weight placement delimiter
        self.parameters_template = self.parameters_template.replace(
        W_WEIGHT_DELIMITER, '')

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
        start = contents.find(START_DELIMITER)
        end = contents.find(END_DELIMITER)

        start += self.START_DELIMITER.len()

        contents = contents[start:end]

        contents = contents.replace(WEIGHT_DATATYPE_DELIMITER, self.WEIGHT_DATATYPE)
        contents = contents.replace(INDEX_DATATYPE_DELIMITER, self.INDEX_DATATYPE)
        #I think there might be more to this than I am thinking
        contents = contents.replace(LAYER_DATATYPE_DELIMITER, self.LAYER_DATATYPE)
        contents = contents.replace(ACTIVATION_DATATYPE_DELIMITER, self.ACTIVATION_DATATYPE)

        return contents
