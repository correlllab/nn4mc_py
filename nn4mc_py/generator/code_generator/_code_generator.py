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
        self.templates = 'templates/' + self.TEMPLATE_TYPE

        self.header_files = {}
        self.source_files = {}

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
            file = LAYER_TEMPLATE_HEADER + layer_type
            with open(self.templates + file + '.template', 'r') as header:
                contents = header.read()
                contents = self.replaceDelimiters(contents)

                self.header_files[file] = contents

            #Replace delimiters and add to layer_templates
            file = LAYER_TEMPLATE_SOURCE + layer_type
            with open(self.templates + file + '.template', 'r') as source:
                contents = source.read()
                contents = self.replaceDelimiters(contents)

                self.source_files[file] = contents

        #For each type scrape and replace delimiters
        #NOTE: Need to change this to grab only the neccessary functions
        file = ACTIVATION_HEADER
        with open(self.templates + file + '.template', 'r') as header:
            contents = header.read()
            head_contents = self.replaceDelimiters(contents)

            activations = ''
            for act_type in activations:
                begin = '<%' + upper(act_type) + '_BEGIN>'
                end = '<%' + upper(act_type) + '_END>'

                start = contents.find(begin) + len(begin)
                stop = contents.find(end)

                activations = activations + contents[start:stop] + '\n'

            head_contents = head_contents.replace(ACTIVATIONS_DELIMITER, activations)

            self.header_files[file] = head_contents

        file = ACTIVATION_SOURCE
        with open(self.templates + file + '.template', 'r') as source:
            contents = source.read()
            head_contents = self.replaceDelimiters(contents)

            activations = ''
            for act_type in activations:
                begin = '<%' + upper(act_type) + '_BEGIN>'
                end = '<%' + upper(act_type) + '_END>'

                start = contents.find(begin) + len(begin)
                stop = contents.find(end)

                activations = activations + contents[start:stop] + '\n'

            head_contents = head_contents.replace(ACTIVATIONS_DELIMITER, activations)

            self.source_files[file] = head_contents

        #Scrape weight file
        file = PARAMETERS_HEADER
        with open(self.templates + file + '.template', 'r') as params:
            contents = params.read()
            contents = self.replaceDelimiters(contents)

            self.header_files[file] = contents

        #Scrape nn4mc main file and add include statements
        file = NEURAL_NETWORK_HEADER
        with open(self.templates + file + '.template', 'r') as header:
            contents = header.read()
            contents = self.replaceDelimiters(contents)

            include_string = ''
            #Might need to be edited
            for layer_type in layers:
                include_string = include_string + '#include ' +\
                                layer_type + '.h\n'

            contents = contents.replace(NN_INCLUDE_DELIMITER, include_string)

            self.header_files[file] = contents

        file = NEURAL_NETWORK_SOURCE
        with open(self.templates + file + '.template', 'r') as source:
            contents = source.read()
            contents = self.replaceDelimiters(contents)

            self.source_files[file] = contents

    # Iterates through graph to extract all metadata and
    # weight data and place in appropriate templates to
    # be dumped.
    #NOTE:
    def processLayers(self):
        #For each node we need to deal with the weights and biases
        #and write the init and fwd functions
        param_template = self.header_files[PARAMETERS_HEADER]
        nn_header = self.header_files[NEURAL_NETWORK_HEADER]
        nn_source = self.source_files[NEURAL_NETWORK_SOURCE]

        for node in self.neural_network.iterate():
            weight_string = node.layer.w.getParams()
            bias_string = node.layer.b.getParams()
            init_string = node.layer.generateInit()
            fwd_string = node.layer.generateFwd()

            #Deal with weights and bias stuff
            param_template = param_template.replace(
                W_WEIGHT_DELIMITER, weight_string + W_WEIGHT_DELIMITER)
            param_template = param_template.replace(
                W_WEIGHT_DELIMITER, bias_string + W_WEIGHT_DELIMITER)

            #Deal with writing the layer
            #HEADER: Add the structs
            pos = nn_header.find(NN_STRUCT_DELIMITER)
            nn_header = nn_header.replace(NN_STRUCT_DELIMITER,
                'struct ' + node.layer.layer_type + ' ' + node.layer.identifier +\
                ';\n' + NN_STRUCT_DELIMITER)

            #SOURCE: Add the init and fwd calls
            pos = nn_source.find(NN_INIT_DELIMITER)
            nn_source = nn_source.replace(NN_INIT_DELIMITER,
                node.layer.identifier + ' = ' + init_string + NN_INIT_DELIMITER)

            pos = nn_source.find(NN_FWD_DELIMITER)
            nn_source = nn_source.replace(NN_FWD_DELIMITER,
                'data = ' + fwd_string + NN_FWD_DELIMITER)

        #Remove the weight placement delimiter
        param_template = param_template.replace(
        W_WEIGHT_DELIMITER, '')
        nn_header = nn_header.replace(NN_STRUCT_DELIMITER, '')
        nn_source = nn_source.replace(NN_INIT_DELIMITER, '')
        nn_source = nn_source.replace(NN_FWD_DELIMITER, '')

        self.header_files[PARAMETERS_HEADER] = param_template
        self.header_files[NEURAL_NETWORK_HEADER] = nn_header
        self.source_files[NEURAL_NETWORK_SOURCE] = nn_source

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
    #NOTE: Done
    def dump(self):
        for header in self.header_files.keys():
            with open(self.output_dir + '/nn4mc' + header, 'a') as outfile:
                outfile.write(self.header_files[header])

        for source in self.source_files.keys():
            with open(self.output_dir + '/nn4mc' + source, 'a') as outfile:
                outfile.write(self.header_files[source])

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
