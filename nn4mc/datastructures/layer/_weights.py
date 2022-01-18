import numpy as np
from nn4mc.datastructures.layer._globals import WEIGHT_DATATYPE
#Class to hold weight or bias arrays for layers
class Weight:
    identifier = None
    values = None
    def __init__(self, id, nparray):
        self.identifier = id #Unique identifier
        self.values = nparray #Numpy array

    def addData(self, nparray):
        self.values = nparray #Numpy array

    #Convert numpy array to string representation
    #for code generation
    #NOTE: Not accounting for different datatypes
    def getParams(self):
        if self.identifier is None:
            return ''
        param_string = ''
        if self.values is not None:
            flat = self.values.flatten()
            size = flat.shape[0]
            param_string = WEIGHT_DATATYPE + ' ' + self.identifier +\
                            '[' + str(size) + '] = {'
            for i in range(size):
                param_string = param_string + str(flat[i])
                if i < size - 1:
                    param_string = param_string + ', '
            param_string = param_string + '};\n'
        return param_string
