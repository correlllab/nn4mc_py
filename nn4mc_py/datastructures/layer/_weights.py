class Weight:
    identifier = ''
    values = None

    def addData(self, nparray):
        self.values = nparray

    #Convert numpy array to string representation
    #NOTE: Not accounting for different datatypes
    def getParams(self):
        if self.values.any() == None:
            return ''

        shape = self.values.shape

        if len(shape) == 1:
            param_string = 'float ' + self.identifier +\
                            '[' + str(shape[0]) + '] = {'

            param_string = param_string + '{'
            for i in range(shape[0]):
                param_string = param_string + str(self.values[i])

                if i<shape[0]-1:
                    param_string = param_string + ', '

            param_string = param_string + '}\n'

        else:
            param_string = 'float ' + self.identifier +\
                            '[' + str(shape[0]) + '][' + str(shape[1]) + '] = {'

            for i in range(shape[0]):
                param_string = param_string + '{'
                for j in range(shape[1]):
                    param_string = param_string + str(self.values[i][j])

                    if j<shape[1]-1:
                        param_string = param_string + ', '

                if i<shape[0]-1:
                    param_string = param_string + '},\n\t\t\t\t\t\t\t\t\t'

            param_string = param_string + '}\n'

        return param_string
