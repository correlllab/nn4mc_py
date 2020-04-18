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
            type = model_layer['class_name']
            name = model_layer['config']['name']
            builder = eval(self.builder_map[type])

            #Build a layer object from metadata
            layer = builder.build_layer(model_layer['config'], name.lower(), type.lower())

            self.nn.addLayer(layer) #Add Layer to neural network
            self.nn.addEdge(last_layer, layer)


            last_layer = layer

    #Parses all of the weights
    #NOTE:
    def parseWeights(self, h5file):
        # with h5py.File(self.file_name,'r') as h5file: #Open file
        weightGroup = h5file['model_weights'] #Open weight group

        input_shape = self.nn_input_size

        for id in weightGroup.keys():
            layer = self.nn.getLayer(id)

            try: #Access weights if they exist
                weight = np.array(weightGroup[id][id]['kernel:0'][()])
                layer.addParameters('weight', (id+'_W', weight))

            except Exception as e: pass#print(e)

            try: #Access biases if they exist
                bias = np.array(weightGroup[id][id]['bias:0'][()])
                layer.addParameters('bias', (id+'_b', bias))

            except Exception as e: pass#print(e)

            try: #Access recurrent weights if they exist
                rec_weight = np.array(weightGroup[id][id]['recurrent_kernel:0'][()])
                layer.addParameters('weight_rec', (id+'_Wrec', weight))

            except Exception as e: pass#print(e)

            print('Computing shape for ', id)
            input_shape = layer.computeOutShape(input_shape)

################################################################################
#Helper functions
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


    #Converts byte array to JSON for scraping
    def bytesToJSON(self, byte_array):
        string = byte_array.decode('utf8')
        JSON = json.loads(string)

        return JSON
