from nn4mc_py.parser._parser import Parser
from ._layerbuilder import *
import h5py

# This class deals with parsing an HDF5 file from a keras neural network model.
# It will scrape the file and generate a NeuralNetwork object
# NOTE:
class HDF5Parser(Parser):
    #Maps layer types to code to dynamically build layers
    builder_map = {'Conv1D' : 'Conv1DBuilder()',
                    'Conv2D' : 'Conv2DBuilder()',
                    'Dense' : 'DenseBuilder()',
                    'Flatten' : 'FlattenBuilder()',
                    'MaxPooling1D' : 'MaxPooling1DBuilder()',
                    'MaxPooling2D' : 'MaxPooling2DBuilder()',
                    'Dropout' : 'DropoutBuilder()',
                    'SimpleRNN' : 'SimpleRNNBuilder()',
                    'GRU' : 'GRUBuilder()',
                    'LSTM' : 'LSTMBuilder()',
                    'Input' : 'InputBuilder()',
                    'Activation' : 'ActivationBuilder()'}

    def __init__(self, file_name):
        self.file_name = file_name #HDF5 model file

        self.layer_builder_list = []
        self.layer_IDs = []
        self.layer_edges = []

        self.layer_map = {}

    #Parses the model and creates a NeuralNetwork
    #NOTE:
    def parse(self):
        parseModelConfig()

        parseWeights()

        callLayerBuilders()

        constructNeuralNetwork()

    #Parses all of the layer metadata
    #NOTE:
    def parseModelConfig(self):
        with h5py.File(self.file_name, 'r') as h5file:
            mainGroup = h5file['/']
            configAttr = mainGroup['model_config']

    #Parses all of the weights
    #NOTE:
    def parseWeights(self):
        with h5py.File(self.file_name,'r') as h5file: #Open file in readonly
            weightGroup = h5file['model_weights'] #Open weight group.

    #
    #NOTE:
    def callLayerBuilders(self):
        pass

    #
    #NOTE:
    def constructNeuralNetwork(self):
        pass

    #
    #NOTE:
    def buildEdges(self):
        pass
