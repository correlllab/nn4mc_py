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
    delim_map = {'LAYER_NAME':'',
    'WEIGHT_NAME':'',
    'BIAS_NAME':'',
    'RECURRENT_WEIGHT_NAME':'',
    'INPUT_SHAPE_0':'',
    'INPUT_SHAPE_1':'',
    'INPUT_SHAPE_2':'',
    'OUTPUT_SIZE':'',
    'KERNEL_SHAPE_0':'',
    'KERNEL_SHAPE_1':'',
    'STRIDE_SHAPE_0':'',
    'STRIDE_SHAPE_1':'',
    'FILTERS':'',
    'PADDING':'',
    'DATA_FORMAT':,
    'DILATION_RATE_0':'',
    'DILATION_RATE_1':'',
    'RECURRENT_ACTIVATION':'',
    'DROPOUT':'',
    'RECURRENT_DROPOUT':'',
    'GO_BACKWARDS':'',
    'POOL_SHAPE_0':'',
    'POOL_SHAPE_1':'',}

    start_delim = '<%'
    end_delim = '>'
