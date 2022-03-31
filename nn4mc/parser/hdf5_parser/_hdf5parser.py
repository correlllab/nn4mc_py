from nn4mc.parser._parser import Parser
from nn4mc.datastructures import NeuralNetwork
from ._layerbuilder import *
import h5py
import numpy as np
import keras
from nn4mc.parser.hdf5_parser.helpers import bytesToJSON
import distutils.spawn
import os

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

    def onnx_parse(self, h5file):
        # Parse model configuration (i.e metadata)
        self.parseModelConfig(h5file)

        # Parse weights and biases
        self.parseWeights(h5file)

        # Close the file
        h5file.close()

    #Parses all of the layer metadata
    #NOTE:
    def parseModelConfig(self, h5file):

        # with h5py.File(self.file_name, 'r') as h5file: #Open hdf5 file
        if not isinstance(h5file, keras.engine.functional.Functional):
            configAttr = h5file['/'].attrs['model_config'] #Gets all metadata
        else:
            configAttr = h5file.to_json()

        configJSON = bytesToJSON(configAttr)

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
        if not isinstance(h5file, keras.engine.functional.Functional):
            weightGroup = h5file['model_weights'] #Open weight group
            # NOTE(sarahaguasvivas) here, the order matters,
            #                       therefore, using different list
        else:
            h5file.save_weights('weights.h5', overwrite = True,
                                        save_format = 'h5')
            # TODO(sarahaguasvivas): find where the dictionary is
            # when you read from h5py.File and assign it to
            # weightGroup,

            """
                NOTE(sarahaguasvivas): What you want at this point is
                for weightGroup to be a serialized dictionary with keys 
                being the names of the layers and values being a tuple
                of weights and biases. For example, 
                
                weightGroup = {'layer_name_0' : (weights_0, biases_0), 
                               'layer_name_1' : (weights_1, biases_1 ...
                               }
                You can either: 
                    1) Try to get it directly from the h5py object that we're 
                    reading --> Running the debug session and digging into the 
                        weightGroup object on the debugger
                    2) Serialize it yourself, but make sure it generalizes 
                    to any model --> digging into model configuration and 
                        h5file.get_weights() to map the layer names to the
                        right weights and biases
            """
            weightGroup = h5py.File('weights.h5', 'r')
        for layer in self.nn.iterate_layer_list():
            id = layer.identifier
            if id in weightGroup.keys() and 'max_pooling1d' not in id \
                    and 'max_pooling2d' not in id and 'flatten' not in id and \
                    'input' not in id:
                # NOTE(sarahaguasvivas): kernel/weight assigment
                gru_keys = [k for k, v in weightGroup[id][id].items() if 'gru_cell' in k]
                if len(gru_keys) > 0:
                    weight = np.array(weightGroup[id][id][gru_keys[0]]['kernel:0'])
                else:
                    weight = np.array(weightGroup[id][id]['kernel:0'][()])
                # NOTE(sarahaguasvivas): bias
                if len(gru_keys) > 0:
                    bias = np.array(weightGroup[id][id][gru_keys[0]]['bias:0'])
                else:
                    bias = np.array(weightGroup[id][id]['bias:0'][()])
                # NOTE(sarahaguasvivas): recurrent weights
                if len(gru_keys) > 0:
                    rec_weight = np.array(weightGroup[id][id][gru_keys[0]]['recurrent_kernel:0'][()])
                else:
                    rec_weight = None
                layer.setParameters('weight', (id + '_W', weight))
                layer.setParameters('bias', (id + '_b', bias))
                layer.setParameters('weight_rec', (id + '_Wrec', rec_weight))

        # NOTE(sarahaguasvivas): calculating output shapes
        input_shape = self.nn_input_size
        for layer in self.nn.iterate_layer_list():
            if "input" not in layer.identifier:
                input_shape = layer.computeOutShape(input_shape)
                print(layer.getParameters())

    #parses model for input size
    def parse_nn_input(self, model_config : dict):
        """
            INPUT: model_config is the json object dictionary
            OUTPUT: numpy array with the input size of the model
        """
        if model_config.get('build_input_shape'):
            self.nn_input_size = model_config['build_input_shape'][1:]
        if model_config['layers'][0].get('config','batch_input_shape'):
            self.nn_input_size = model_config['layers'][0]['config']['batch_input_shape'][1:]