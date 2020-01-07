import h5py

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
