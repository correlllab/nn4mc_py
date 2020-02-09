class Weight:
    identifier = ''
    values = None

    def addData(self, nparray):
        self.values = nparray

    #Convert numpy array to string representation
    #NOTE: Not accounting for different datatypes
    def getParams():
        #const <%WEIGHT_DATATYPE_DELIMITER> <%WEIGHT_NAME><%WEIGHT_INDEX> = <%WEIGHT_DATA>;
        shape = self.values.shape
        param_string = 'float ' + self.identifier
                        + '[' + shape[0] + '][' + shape[1] + '] = {'

        for i in range(shape[0]):
            param_string = param_string + '{'
            for j in range(shape[1]):

                if j<shape[1]-1:
                    param_string = param_string + ', '

            if i<shape[0]-1:
                param_string = param_string + '},\n'

        param_string = param_string + '}\n'

        return param_string
