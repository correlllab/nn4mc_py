import numpy as np

#Class to hold weight or bias arrays for layers
class Weight:
    def __init__(self, id, nparray=None):
        self.identifier = id #Unique identifier
        self.values = nparray #Numpy array

    def addData(self, nparray):
        self.values = nparray #Numpy array

    #Convert numpy array to string representation
    #for code generation
    #NOTE: Not accounting for different datatypes
    def getParams(self):
        if self.values.any() == None:
            return ''

        flat = self.values.flatten()
        size = flat.shape[0]

        param_string = 'float ' + self.identifier +\
                        '[' + str(size) + '] = {'

        for i in range(size):
            param_string = param_string + str(flat[i])

            if i<size-1:
                param_string = param_string + ', '

        param_string = param_string + '}\n\n'

        return param_string


    #Deprecated
    # def getParams(self):
    #     if self.values.any() == None:
    #         return ''
    #
    #     shape = self.values.shape
    #
    #     if len(shape) == 1:
    #         param_string = 'float ' + self.identifier +\
    #                         '[' + str(shape[0]) + '] = {'
    #
    #         for i in range(shape[0]):
    #             param_string = param_string + str(self.values[i])
    #
    #             if i<shape[0]-1:
    #                 param_string = param_string + ', '
    #
    #         param_string = param_string + '}\n'
    #
    #     else:
    #         param_string = 'float ' + self.identifier +\
    #                         '[' + str(shape[0]) + '][' + str(shape[1]) + '] = {'
    #
    #         for i in range(shape[0]):
    #             param_string = param_string + '{'
    #             for j in range(shape[1]):
    #                 param_string = param_string + str(self.values[i][j])
    #
    #                 if j<shape[1]-1:
    #                     param_string = param_string + ', '
    #
    #             if i<shape[0]-1:
    #                 param_string = param_string + '},\n\t\t\t\t\t\t\t\t\t'
    #
    #         param_string = param_string + '}\n'
    #
    #     return param_string
