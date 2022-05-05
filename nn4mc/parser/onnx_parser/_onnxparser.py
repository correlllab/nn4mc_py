from nn4mc.parser._parser import Parser
from nn4mc.datastructures import NeuralNetwork
# from ._layerbuilder import *
import h5py
import onnx
import keras
# from onnx2keras import onnx_to_keras
# from nn4mc.parser.onnx_parser.onnx_helpers import HDF5Parser
import numpy as np
from nn4mc.parser.onnx_parser.onnx_helpers import onnx2keras
from nn4mc.parser.hdf5_parser._hdf5parser import HDF5Parser
from tensorflow import keras

class ONNXParser(Parser):

    def __init__(self, file):
        self.file = file
        self.nn = NeuralNetwork()
        self.nn_input_size = None

    def parse(self):
        h5format = onnx2keras(self.file)

        onnx_model = onnx.load(self.file)
        # print(type(onnx_model))

        h5parser = HDF5Parser(h5format)
        h5parser.file = self.file
        h5parser.onnx_parse(h5format)
        self.nn = h5parser.nn

        self.parseModelConfig(onnx_model)

        # parse weights and biases
        self.parseWeights(onnx_model)

        # close the file
        # h5format.close()

    def parseModelConfig(self, h5file):

        # with h5py.File(self.file_name, 'r') as h5file: #Open hdf5 file
        #if not isinstance(h5file, keras.engine.functional.Functional):
        configAttr = h5file['/'].attrs['model_config'] #Gets all metadata
        #else:
        #    configAttr = h5file.to_json()

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

    def _parseONNX(self):
        return self.parse.h5format

    def parseWeights(self, h5file, _parseONNX = True):

        weightGroup = h5file.graph.initializer

        if (not _parseONNX):
            pass

        else:
            for layer in self.nn.iterate_layer_list():

                id = layer.identifier

                if id in weightGroup and 'max_pooling1d' not in id \
                        and 'max_pooling2d' not in id and 'flatten' not in id and \
                        'input' not in id:

                    gru_keys = [k for k, v in weightGroup[id][id].items() if 'gru_cell' in k]
                    # kernel/weight assignment
                    if len(gru_keys) > 0:
                        weight = np.array(weightGroup[id][id][gru_keys[0]]['kernel:0'])
                    else:
                        weight = np.array(weightGroup[id][id]['kernel:0'][()])
                    # bias
                    if len(gru_keys) > 0:
                        bias = np.array(weightGroup[id][id][gru_keys[0]]['bias:0'])
                    else:
                        bias = np.array(weightGroup[id][id]['bias:0'][()])
                    # weight
                    if len(gru_keys) > 0:
                        rec_weight = np.array(weightGroup[id][id][gru_keys[0]]['recurrent_kernel:0'][()])
                    else:
                        rec_weight = None

                    layer.setParameters('weight', (id + '_W', weight))
                    layer.setParameters('bias', (id + '_b', bias))
                    layer.setParameters('weight_rec', (id + '_Wrec', rec_weight))

            input_shape = self.nn_input_size
            # print('aaaaa', input_shape)
            for layer in self.nn.iterate_layer_list():
                if "input" not in layer.identifier:
                    input_shape = layer.computeOutShape(input_shape)
                    print(layer.getParameters())

    def parse_nn_input(self, model_config : dict):
        """
            INPUT: model_config is the json object dictionary
            OUTPUT: numpy array with the input size of the model
        """
        if model_config.get('build_input_shape'):
            self.nn_input_size = model_config['build_input_shape'][1:]
        if model_config['layers'][0].get('config','batch_input_shape'):
            self.nn_input_size = model_config['layers'][0]['config']