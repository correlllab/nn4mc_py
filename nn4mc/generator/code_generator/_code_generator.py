from nn4mc.datastructures import NeuralNetwork
from ._globals import G, layer_type_lookup
import numpy as np
import os, json

#This class deals with iterating on a NerualNetwork object and
#generating C code.
#NOTE:
class Generator():
    #NOTE: The following constants should actually be variable in the future
    #more customization in terms of these values.
    INDEX_DATATYPE = 'int'
    LAYER_OUTPUT_DATATYPE = 'float'
    ACTIVATION_DATATYPE = 'char'
    WEIGHT_DATATYPE = 'const float'

    #NOTE:
    def __init__(self, nn_obj, template_type='c_standard'):
        self.nn = nn_obj #NerualNetwork object to iterate on
        self.TEMPLATE_TYPE = template_type #Templates

        path = os.path.dirname(__file__)
        self.templates_path = os.path.join(path, 'templates/' + self.TEMPLATE_TYPE)

        self.header_files = {}
        self.source_files = {}
        self.init_strings = {}
        self.fwd_strings = {}

    # Generates the code
    #NOTE: F specifies file output, and W specifies web output
    def generate(self, output_dir='', output_type='F'):
        self.processTemplates() #Processes required templates

        self.processLayers() #Processes layers

        if output_type == 'F':
            self.buildFileTree(output_dir) #Builds output file directory
            self.dump(output_dir) #Dumps output code

        elif output_type == 'D':
            file_data = {**self.header_files, **self.source_files}
            return file_data

        else: pass #Raise an error

    # Iterates through graph to extract which layers and
    # activations are required. Also replaces any delimiters
    # that can be at this point
    #NOTE:
    def processTemplates(self):
        layers = []
        activations = []

        #Get lists of all activations and layers
        for node in self.nn.iterate():
            type = node.layer.layer_type
            if type != 'input' and type != 'inputlayer' and type != 'flatten':
                if hasattr('node.layer', 'activation'):
                    activation = node.layer.activation
                    if activation != '' and activation not in activations and activation != 'linear':
                        activations.append(activation)
                if type not in layers:
                    layers.append(type)
                    
        #For each layer type scrape file and replace delimiters
        for layer_type in layers:
            #Replace delimiters and add to layer_templates
            file = G.LAYER_TEMPLATE_HEADER + layer_type + '.h'
            with open(self.templates_path + file + '.template', 'r') as header:
                contents = header.read()
                contents = self.replaceDelimiters(contents)

                self.header_files[file] = contents

            #Replace delimiters and add to layer_templates
            file = G.LAYER_TEMPLATE_SOURCE + layer_type + '.cpp'
            with open(self.templates_path + file + '.template', 'r') as source:
                contents = source.read()
                strings = self.extractStrings(contents)
                contents = self.replaceDelimiters(contents)

                self.source_files[file] = contents
                self.init_strings[layer_type] = strings[0]
                self.fwd_strings[layer_type] = strings[1]

        #For each activation type scrape and replace delimiters
        file = G.ACTIVATION_HEADER
        with open(self.templates_path + file + '.template', 'r') as header:
            contents = header.read()
            head_contents = self.replaceDelimiters(contents)

            activations_string = ''
            for act_type in activations:
                begin = '<%' + act_type.upper() + '_BEGIN>'
                end = '<%' + act_type.upper() + '_END>'

                start = contents.find(begin) + len(begin)
                stop = contents.find(end)
                activations_string = activations_string + contents[start:stop] + '\n'

            activations_string = self.replaceDelimiters(activations_string)
            head_contents = head_contents.replace(G.ACTIVATIONS_DELIMITER, activations_string)

            self.header_files[file] = head_contents

        file = G.ACTIVATION_SOURCE
        with open(self.templates_path + file + '.template', 'r') as source:
            contents = source.read()
            head_contents = self.replaceDelimiters(contents)

            activations_string = ''
            for act_type in activations:
                begin = '<%' + act_type.upper() + '_BEGIN>'
                end = '<%' + act_type.upper() + '_END>'

                start = contents.find(begin) + len(begin)
                stop = contents.find(end)
                activations_string = activations_string + contents[start:stop] + '\n'
            activations_string = self.replaceDelimiters(activations_string)
            head_contents = head_contents.replace(G.ACTIVATIONS_DELIMITER, activations_string)

            self.source_files[file] = head_contents

        #Scrape weight file
        file = G.PARAMETERS_HEADER
        with open(self.templates_path + file + '.template', 'r') as params:
            contents = params.read()
            contents = self.replaceDelimiters(contents)

            self.header_files[file] = contents

        #Scrape nn4mc main include file and add include statements
        file = G.NEURAL_NETWORK_HEADER
        with open(self.templates_path + file + '.template', 'r') as header:
            contents = header.read()
            contents = self.replaceDelimiters(contents)
            #contents.replace(G.NN_INCLUDE_DELIMITER, include_string)
            self.header_files[file] = contents

        #Scrape nn4mc main source file
        file = G.NEURAL_NETWORK_SOURCE
        with open(self.templates_path + file + '.template', 'r') as source:
            contents = source.read()
            contents = self.replaceDelimiters(contents)
            include_string = ''
            # Might need to be edited
            for layer_type in layers:
                include_string = include_string + '#include ' + \
                                 layer_type + '.h\n'
            contents.replace(G.NN_INCLUDE_DELIMITER, include_string)
            self.source_files[file] = contents
    # Iterates through graph to extract all metadata and
    # weight data and place in appropriate templates to
    # be dumped.
    #NOTE:
    def processLayers(self):
        #For each node we need to deal with the weights and biases
        #and write the init and fwd functions
        param_template = self.header_files[G.PARAMETERS_HEADER]
        nn_header = self.header_files[G.NEURAL_NETWORK_HEADER]
        nn_source = self.source_files[G.NEURAL_NETWORK_SOURCE]
        layers = []
        activations = []
        for node in self.nn.iterate():
            type = node.layer.layer_type
            if type != 'input' and type != 'inputlayer' and type != 'flatten':
                if hasattr('node.layer', 'activation'):
                    activation = node.layer.activation
                    if activation != '' and activation not in activations and activation != 'linear':
                        activations.append(activation)
                if type not in layers:
                    layers.append(type)
        include_string = ''
        # Might need to be edited
        for layer_type in layers:
            include_string = include_string + '#include ' + \
                             layer_type + '.h\n'
        for node in self.nn.iterate():
            if node.layer.layer_type != 'input' and \
                    node.layer.layer_type != 'flatten' and \
                    node.layer.layer_type != 'inputlayer':
                param_string = node.layer.getParameters()
                init_string = node.layer.generateCall(self.init_strings[node.layer.layer_type])
                fwd_string = node.layer.generateCall(self.fwd_strings[node.layer.layer_type])
                act_string = node.layer.generateAct()

                #Deal with weights and bias stuff
                param_template = param_template.replace(
                    G.W_WEIGHT_DELIMITER, param_string + G.W_WEIGHT_DELIMITER)

                #Deal with writing the layer
                #HEADER: Add the structs
                pos = nn_source.find(G.NN_STRUCT_DELIMITER)
                nn_source = nn_source.replace(G.NN_STRUCT_DELIMITER,
                    "struct " + layer_type_lookup[node.layer.layer_type] + ' ' + node.layer.identifier + \
                    ';\n' + G.NN_STRUCT_DELIMITER)
                nn_source = nn_source.replace(G.NN_INCLUDE_DELIMITER,
                                             include_string)

                #SOURCE: Add the init and fwd calls
                pos = nn_source.find(G.NN_INIT_DELIMITER)
                nn_source = nn_source.replace(G.NN_INIT_DELIMITER,
                    init_string + G.NN_INIT_DELIMITER)

                pos = nn_source.find(G.NN_FWD_DELIMITER)
                nn_source = nn_source.replace(G.NN_FWD_DELIMITER,
                    fwd_string + G.NN_FWD_DELIMITER)

                #Add the activation funcion
                pos = nn_source.find(G.NN_FWD_DELIMITER)
                nn_source = nn_source.replace(G.NN_FWD_DELIMITER,
                    act_string + G.NN_FWD_DELIMITER)

        #Remove delimiters
        param_template = param_template.replace(
        G.W_WEIGHT_DELIMITER, '')
        nn_source = nn_source.replace(G.NN_STRUCT_DELIMITER, '')
        nn_source = nn_source.replace(G.NN_INIT_DELIMITER, '')
        nn_source = nn_source.replace(G.NN_FWD_DELIMITER, '')

        self.header_files[G.PARAMETERS_HEADER] = param_template
        self.header_files[G.NEURAL_NETWORK_HEADER] = nn_header
        self.source_files[G.NEURAL_NETWORK_SOURCE] = nn_source

    # Builds the output file structure
    #NOTE: Need to add more error handling
    def buildFileTree(self, output_dir):
        directories = []
        directories.append(output_dir + '/nn4mc')
        directories.append(output_dir + '/nn4mc/include')
        directories.append(output_dir + '/nn4mc/include/layers')
        directories.append(output_dir + '/nn4mc/src')
        directories.append(output_dir + '/nn4mc/src/layers')
        try:
            for dir in directories:
                os.mkdir(dir)
        except Exception as e:
            print('GENERATOR: Output directory exists, moving on.')

    # Dumps all files into output structure
    #NOTE: Done
    def dump(self, output_dir):
        for header in self.header_files.keys():
            with open(output_dir + '/nn4mc' + header, 'w') as outfile:
                outfile.write(self.header_files[header])

        for source in self.source_files.keys():
            with open(output_dir + '/nn4mc' + source, 'w') as outfile:
                outfile.write(self.source_files[source])

################################################################################
#Helper Functions
    # Looks for all standard delimiters and replaces them with actual values
    #NOTE:
    def replaceDelimiters(self, contents):
        start = contents.find(G.START_DELIMITER)
        end = contents.find(G.END_DELIMITER)

        if start != -1:
            start += len(G.START_DELIMITER)

            contents = contents[start:end]

        contents = contents.replace(G.WEIGHT_DATATYPE_DELIMITER, self.WEIGHT_DATATYPE)
        contents = contents.replace(G.INDEX_DATATYPE_DELIMITER, self.INDEX_DATATYPE)
        #I think there might be more to this than I am thinking
        contents = contents.replace(G.LAYER_DATATYPE_DELIMITER, self.LAYER_OUTPUT_DATATYPE)
        contents = contents.replace(G.ACTIVATION_DATATYPE_DELIMITER, self.ACTIVATION_DATATYPE)

        return contents

    #Looks for initialization and forward strings and extracts them
    def extractStrings(self, contents):
        init_string = None
        fwd_string = None

        start = contents.find(G.START_INIT)
        end = contents.find(G.END_INIT)

        if start != -1:
            start += len(G.START_INIT)

            init_string = contents[start:end]

        start = contents.find(G.START_FWD)
        end = contents.find(G.END_FWD)

        if start != -1:
            start += len(G.START_FWD)

            fwd_string = contents[start:end]

        return (init_string, fwd_string)
