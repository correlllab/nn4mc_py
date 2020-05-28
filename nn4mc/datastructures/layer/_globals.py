#Contains all global maps for Layer
class G():
    #Padding
    padding = {'valid':'0x00',
                'casual':'0x02',
                'same':'0x03'}

    #Dataformat
    dataformat = {'channels_last':'0x00',
                    'channels_first':'0x02'}

    #Delimiters
    start_delim = '<%'
    end_delim = '>'

    delim_map = {'LAYER_NAME':'self.identifier',
    'WEIGHT_NAME':'self.params["w"].identifier',
    'BIAS_NAME':'self.params["b"].identifier',
    'RECURRENT_WEIGHT_NAME':'self.params["w_rec"].identifier',
    'INPUT_SHAPE_0':'str(self.input_shape[0])',
    'INPUT_SHAPE_1':'str(self.input_shape[1])',
    'INPUT_SHAPE_2':'str(self.input_shape[2])',
    'OUTPUT_SHAPE':'str(self.output_shape[0])',
    'KERNEL_SHAPE_0':'str(self.kernel_shape[0])',
    'KERNEL_SHAPE_1':'str(self.kernel_shape[1])',
    'STRIDES_0':'str(self.strides[0])',
    'STRIDES_1':'str(self.strides[0])',
    'FILTERS':'str(self.filters)',
    'PADDING':'G.padding[self.padding]',
    'DATA_FORMAT':'G.dataformat[self.data_format]',
    'DILATION_RATE_0':'str(self.dilation_rate[0]) ',
    'DILATION_RATE_1':'str(self.dilation_rate[1]) ',
    'RECURRENT_ACTIVATION':'self.recurrent_activation',
    'DROPOUT':'str(self.dropout)',
    'RECURRENT_DROPOUT':'str(self.recurrent_dropout)',
    'GO_BACKWARDS':'self.go_backwards',
    'POOL_SHAPE_0':'str(self.pool_shape[0])',
    'POOL_SHAPE_1':'str(self.pool_shape[1])'}
