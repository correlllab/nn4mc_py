from nn4mc.parser._parser import Parser
from nn4mc.datastructures import NeuralNetwork
from ._layerbuilder import *
import h5py
import numpy as np
import json

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
                    'InputLayer': 'InputBuilder()',
                    'Activation' : 'ActivationBuilder()'}

    #NOTE: file is etheir a file name or a filetype object
    #If file object it needs to be binary I/O i.e 'rb'
    def __init__(self, file):
        self.file = file #HDF5 model file
        self.nn = NeuralNetwork() #NeuralNetwork to be filled
        self.nn_input_size = None

    #Parses the model and creates a NeuralNetwork
    #NOTE:
    def parse(self):
        #Open file if not already file object
        h5file = h5py.File(self.file, 'r')

        #Parse model configuration (i.e metadata)
        self.parseModelConfig(h5file)

        #Parse weights and biases
        self.parseWeights(h5file)

        #Close the file
        h5file.close()

    #Parses all of the layer metadata
    #NOTE:
    def parseModelConfig(self, h5file):
        # with h5py.File(self.file_name, 'r') as h5file: #Open hdf5 file
        configAttr = h5file['/'].attrs['model_config'] #Gets all metadata
        configJSON = self.bytesToJSON(configAttr)

        self.parse_nn_input(configJSON['config'])

        #This adds an input layer before everything, not sure if it is
        #really neccessary.
        #NOTE: Determine if this is neccessary
        last_layer = Input('input_1','input')
        self.nn.addLayer(last_layer)

        #NOTE: Could check to see if its sequential here
        for model_layer in configJSON['config']['layers']:
            type_ = model_layer['class_name']
            name = model_layer['config']['name']

            if type_ in self.builder_map.keys():
                builder = eval(self.builder_map[type_])

                #Build a layer object from metadata
                layer = builder.build_layer(model_layer['config'], name.lower(), type_.lower())

                self.nn.addLayer(layer) #Add Layer to neural network
                self.nn.addEdge(last_layer, layer)

                last_layer = layer

    #Parses all of the weights
    #NOTE:
    def parseWeights(self, h5file):
        weightGroup = h5file['model_weights'] #Open weight group
        print(weightGroup)
        input_shape = self.nn_input_size

        # NOTE(sarahaguasvivas) here, the order matters,
        #                       therefore, using different list
        for id in weightGroup.keys():
            if id != 'top_level_model_weights':
                layer = self.nn.getLayer(id)

                try: #Access weights if they exist
                    weight = np.array(weightGroup[id][id]['kernel:0'][()])
                    layer.addParameters('weight', (id+'_W', weight))

                except Exception as e: pass#print(e)

                try: #Access biases if they exist
                    bias = np.array(weightGroup[id][id]['bias:0'][()])
                    layer.addParameters('bias', (id+'_b', bias))

                except Exception as e: pass#print(e)
                print(id, weightGroup[id][id].keys())
                try: #Access recurrent weights if they exist
                    rec_weight = np.array(weightGroup[id]['kernel:1'][()])
                    print(rec_weight)
                    layer.addParameters('weight_rec', (id+'_Wrec', rec_weight))

                except Exception as e: pass#print(e)

        for layer in self.nn.layer_list:
            input_shape = layer.computeOutShape(input_shape)

    #Parses model for input size
    def parse_nn_input(self, model_config : dict):
        """
            INPUT: model_config is the json object dictionary
            OUTPUT: numpy array with the input size of the model
        """
        if model_config.get('build_input_shape'):
            self.nn_input_size = model_config['build_input_shape'][1:]
        if model_config['layers'][0].get('config','batch_input_shape'):
            self.nn_input_size = model_config['layers'][0]['config']['batch_input_shape'][1:]

################################################################################
#Helper functions
    #Converts byte array to JSON for scraping
    def bytesToJSON(self, byte_array):
        print(byte_array)
        string = bytearray(byte_array).decode('utf8')
        JSON = json.loads(string)
        return JSON
