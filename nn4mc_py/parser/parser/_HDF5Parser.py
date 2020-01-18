from ._parser import Parser
import h5py

class Opdata:
    pass
    #Maybe declare vaiables here or maybe not.

class OpdataWeights:
    pass
    #Maybe declare variables here or maybe not.

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
        with h5py.File(self.file_name, 'r') as h5file:
            mainGroup = h5file['/']
            configAttr = mainGroup['model_config']

    def parseWeights(self):
        with h5py.File(self.file_name,'r') as h5file: #Open file in readonly
            weightGroup = h5file['model_weights'] #Open weight group.


    def constructBuilderMap(self):
        pass

    def callLayerBuilders(self):
        pass

    def buildEdges(self):
        pass

    def constructNeuralNetwork(self):
        pass
