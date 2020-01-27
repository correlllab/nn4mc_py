from nn4mc_py.datastructures import *
import numpy as np
import os

class Generator():

    def __init__(self, nn_obj):
        pass

    # Generates the code
    #NOTE:
    def generate(self):
        self.buildFileTree()

        self.processTemplates()

        self.processLayers()

        self.dump()

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
    #NOTE:
    def buildFileTree(self):
        pass

    # Dumps all files into output structure
    #NOTE:
    def dump(self):
        pass
