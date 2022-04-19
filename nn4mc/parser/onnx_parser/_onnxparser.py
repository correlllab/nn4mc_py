from nn4mc.parser._parser import Parser
from nn4mc.datastructures import NeuralNetwork
# from ._layerbuilder import *
# import onnx
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


        h5parser = HDF5Parser(h5format)
        h5parser.file = self.file
        h5parser.onnx_parse(h5format)
        self.nn = h5parser.nn

        # parse weights and biases
        self.parseWeights(h5format)

        # close the file
        # h5format.close()

    def _parseONNX(self):
        return self.parse.h5format

    def parseWeights(self, h5file, _parseONNX = False):

        weightGroup = h5file['model_weights']

        if (not _parseONNX):

            for layer in self.nn.iterate_layer_list():

                id = layer.identifier

                if id in weightGroup.keys() and 'max_pooling1d' not in id \
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
            for layer in self.nn.iterate_layer_list():
                if "input" not in layer.identifier:
                    input_shape = layer.computeOutShape(input_shape)
                    print(layer.getParameters())
        else:
            pass