from ._parser import Parser
import h5py

class HDF5Parser(Parser):
    def __init__(self, file_name):
        self.file_name = file_name

        self.layer_builder_list = []
        self.layer_IDs = []
        self.layer_edges = []

        self.builder_map = {}
        self.layer_map = {}

    def parse(self):
        pass

    def parseModelConfig(self):
        with h5py.File(self.file_name, 'r') as h5File:
            mainGroup = h5File['/']
            configAttr = mainGroup['model_config']

    def parseWeights(self):
        with h5py.File(self.file_name,'r') as h5File: #Open file in readonly
            weightGroup = h5File['model_weights'] #Open weight group.


    def constructBuilderMap(self):
        pass

    def callLayerBuilders(self):
        pass

    def buildEdges(self):
        pass

    def constructNeuralNetwork(self):
        pass
