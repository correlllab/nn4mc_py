from nn4mc_py.parser._parser import Parser
from nn4mc_py.datastructures import NeuralNetwork, Layer
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

        self.nn = NeuralNetwork()

    #Parses the model and creates a NeuralNetwork
    #NOTE:
    def parse(self):
        parseModelConfig()

        parseWeights()

        constructNeuralNetwork()

    #Parses all of the layer metadata
    #NOTE:
    def parseModelConfig(self):
        with h5py.File(self.file_name, 'r') as h5file: #Open file
            configAttr = h5file['/'].attrs['model_config'] #Gets all metadata

            #NOTE: Could check to see if its sequential here
            for model_layer in configAttr['config']['layers']:
                type = model_layer['class_name']
                name = model_layer['name']
                builder = eval(self.builder_map[type])

                #Build a layer object from metadata
                layer = builder.build_layer(model_layer, name, type)

                self.nn.addLayer(layer) #Add Layer to neural network

                #NOTE: Could do edges here as well looking back 1 step


    #Parses all of the weights
    #NOTE:
    def parseWeights(self):
        with h5py.File(self.file_name,'r') as h5file: #Open file
            weightGroup = h5file['model_weights'] #Open weight group

            for id in weightGroup.keys():
                try: #Access weights and biases
                    #NOTE: Not a numpy array, but hdf5 dataset is similar
                    weight = weightGroup[id][id]['kernel:0']
                    bias = weightGroup[id][id]['bias:0']

                except: #Layer type with not weights
                    weight = None
                    bias = None

                #Do something here

    #
    #NOTE:
    def constructNeuralNetwork(self):
        pass
