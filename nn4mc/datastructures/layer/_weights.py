import numpy as np

#Class to hold weight or bias arrays for layers
class Weight:
    def __init__(self, id, nparray):
        self.identifier = id #Unique identifier
        self.values = nparray #Numpy array

    def addData(self, nparray):
        self.values = nparray #Numpy array

    #Convert numpy array to string representation
    #for code generation
    #NOTE: Not accounting for different datatypes
    def getParams(self):
        if self.identifier==None:
            return ''

        flat = self.values.flatten()
        size = flat.shape[0]

        #This is where you would account for datatypes
        param_string = 'float ' + self.identifier +\
                        '[' + str(size) + '] = {'

        for i in range(size):
            param_string = param_string + str(flat[i])

            if i<size-1:
                param_string = param_string + ', '

        param_string = param_string + '}\n\n'

        return param_string
