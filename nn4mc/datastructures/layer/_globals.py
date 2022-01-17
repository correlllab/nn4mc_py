#Contains all global maps for Layer

INDEX_DATATYPE = 'int'
LAYER_OUTPUT_DATATYPE = 'float'
ACTIVATION_DATATYPE = 'char'
WEIGHT_DATATYPE = 'const float'

class G():
    #Padding
    padding = {
        'valid':'0x00',
        'casual':'0x02',
        'same':'0x03'
    }

    #Dataformat
    dataformat = {
        'channels_last':'0x00',
        'channels_first':'0x02'
    }

    #Delimiters
    start_delim = '<%'
    end_delim = '>'

    delim_map = {
        'ACTIVATION':'self.activation',
        'BIAS_NAME':'self.params["b"].identifier',
        'DATA_FORMAT':'G.dataformat[self.data_format]',
        'DILATION_RATE_0':'str(self.dilation_rate[0]) ',
        'DILATION_RATE_1':'str(self.dilation_rate[1]) ',
        'DROPOUT':'str(self.dropout)',
        'FILTERS':'str(self.filters)',
        'GO_BACKWARDS':'self.go_backwards',
        'INPUT_SHAPE_0':'str(self.input_shape[0])',
        'INPUT_SHAPE_1':'str(self.input_shape[1])',
        'INPUT_SHAPE_2':'str(self.input_shape[2])',
        'KERNEL_SHAPE_0':'str(self.kernel_shape[0])',
        'KERNEL_SHAPE_1':'str(self.kernel_shape[1])',
        'LAYER_NAME':'self.identifier',
        'OUTPUT_SHAPE':'str(self.output_shape[0])',
        'PADDING':'G.padding[self.padding]',
        'POOL_SHAPE_0':'str(self.pool_shape[0])',
        'POOL_SHAPE_1':'str(self.pool_shape[1])',
        'RECURRENT_ACTIVATION':'self.recurrent_activation',
        'RECURRENT_DROPOUT':'str(self.recurrent_dropout)',
        'RECURRENT_WEIGHT_NAME':'self.params["w_rec"].identifier',
        'STRIDES_0':'str(self.strides[0])',
        'STRIDES_1':'str(self.strides[0])',
        'USE_BIAS':'self.use_bias',
        'WEIGHT_NAME':'self.params["w"].identifier',
    }
