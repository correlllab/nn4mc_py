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

        self.layer_templates = []
        self.activation_functions = []

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
        for node in self.neural_network.iterate():
            pass

    # Iterates through graph to extract all metadata and
    # weight data and place in appropriate templates to
    # be dumped.
    #NOTE:
    def processLayers(self):
        for node in self.neural_network.iterate():
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
