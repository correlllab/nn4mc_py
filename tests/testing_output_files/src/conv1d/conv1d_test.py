import conv1d
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.backend import clear_session
import numpy as np
import unittest
from typing import List, Final
import ctypes
import copy

def swig_py_object_2_list(object, size : int) -> List[float]:
    """
        Converts SwigPyObject to List[float]
    """
    y = (ctypes.c_float * size).from_address(int(object))
    new_object = []
    for i in range(size):
        new_object += [float(y[i])]
    return new_object

def swig_py_object_2_list_int(object, size : int) -> List[int]:
    """
        Converts SwigPyObject to List[float]
    """
    y = (ctypes.c_float * size).from_address(int(object))
    new_object = []
    for i in range(size):
        new_object += [int(y[i])]
    return new_object

def list_2_swig_float_pointer(list : List[float], size : int):
    """
        Converts from list of floats to swig float* object
    """
    test_buffer = conv1d.input(size)
    for i in range(size):
        test_buffer[i] = float(list[i])
    return test_buffer


activation_dictionary = {'softmax': 0x00,
                         'elu': 0x02, 'selu': 0x03,
                         'softplus': 0x04, 'softsign': 0x05,
                         'relu': 0x06, 'tanh': 0x07, 'sigmoid': 0x08,
                         'hard_sigmoid': 0x09, 'exponential': 0xA, 'linear': 0xB,
                         'custom': 0xC}

padding_dictionary = {'valid': 0x00, 'causal': 0x02, 'same': 0x03}

dataformat_dictionary = {'channels_last': 0x00, 'channels_first': 0x02}

class Conv1DTest(unittest.TestCase):
    """
        Conv1D
    """
    def __generate_sample(self, input_dims):
        return np.random.normal(0.0, 20, size = input_dims)

    def __keras_build(self, build_dict : dict):
        model = Sequential()
        model.add(Conv1D(
                    input_shape = build_dict['input_shape'],
                    filters = build_dict['filters'],
                    kernel_size = build_dict['kernel_size'],
                    strides = build_dict['strides'],
                    padding = build_dict['padding'],
                    data_format = build_dict['data_format'],
                    dilation_rate =  build_dict['dilation_rate'],
                    activation = build_dict['activation'],
                    use_bias = build_dict['use_bias']
                    ))
        model.trainable = False
        return model

    def test_padding(self):
        shape = np.random.randint(3, size=2).tolist()
        input_dims = (1, shape[0] + 1, shape[1] + 1)
        input_ = self.__generate_sample(input_dims)

        build_dict = {'filters': 32, 'kernel_size': 3, 'strides': 1, 'padding': 'valid',
                      'data_format': 'channels_last', 'dilation_rate': 1, 'activation': 'linear',
                      'use_bias': True, 'input_shape' : input_dims}

        weight = conv1d.input(input_dims[1] * input_dims[2])
        bias = conv1d.input(input_dims[2])

        padding = [0x00, 0x02, 0x03]
        input_ = input_.flatten().tolist()
        input_all = list_2_swig_float_pointer(input_, input_dims[1]*input_dims[2])
        for pad in padding:
            input = copy.copy(input_all)
            layer = conv1d.build_layer_conv1d(weight.cast(), bias.cast(),
                                              build_dict['kernel_size'], build_dict['strides'],
                                              input_dims[1], input_dims[2], build_dict['filters'],
                                              activation_dictionary[build_dict['activation']],
                                              pad,
                                              dataformat_dictionary[build_dict['data_format']],
                                              build_dict['dilation_rate'])
            if (pad == 0x00):
                new_size =  len(input_)
            if (pad == 0x02):
                left_pad = build_dict['dilation_rate'] * (build_dict['kernel_size'] - 1)
                new_size = len(input_) + input_dims[1]*left_pad
            if (pad == 0x03):
                pad = build_dict['filters'] // 2
                new_size = len(input_) + input_dims[1]*pad

            padding_result = conv1d.padding_1d(layer, input.cast())
            padding_result = swig_py_object_2_list(padding_result, new_size)

    def __c_fwd(self, build_dict : dict, input_, weight, bias, weight_size, bias_size, input_dims):
        weight = list_2_swig_float_pointer(weight, weight_size)
        bias = list_2_swig_float_pointer(bias, bias_size)
        input_length = input_.size


        input_ = input_.flatten().tolist()
        input_all = list_2_swig_float_pointer(input_, len(input_))

        output_dims = build_dict['filters'] * ((input_dims[1] - \
                      build_dict['kernel_size']) // build_dict['strides'] + 1)

        layer = conv1d.build_layer_conv1d(weight.cast(), bias.cast(),
                                              build_dict['kernel_size'], build_dict['strides'],
                                              input_dims[1], input_dims[2], build_dict['filters'],
                                              activation_dictionary[build_dict['activation']],
                                              padding_dictionary[build_dict['padding']],
                                              dataformat_dictionary[build_dict['data_format']],
                                              build_dict['dilation_rate'])

        output = conv1d.fwd_conv1d(layer, input_all.cast())
        output = swig_py_object_2_list(output, output_dims)
        return output, output_dims

    def __keras_fwd(self, config_dict : dict, input_, weight, bias):
        model = self.__keras_build(config_dict)
        model.set_weights([weight, bias])
        prediction = model.predict(input_)
        del model
        clear_session()
        return prediction

    def test_fwd(self):
        N = 1000
        for _ in range(N):
            print(_)
            strides = int(np.random.randint(1, 5, size = 1)[0])
            print(strides)
            build_dict = {'filters': 32, 'kernel_size' : 3, 'strides' : strides, 'padding' : 'valid',
                    'data_format' : 'channels_last', 'dilation_rate' : 1, 'activation' : 'relu',
                    'use_bias' : True}

            shape = np.random.randint(build_dict['kernel_size'], 5, size = 2).tolist()
            input_dims = (1, shape[0] + 1, shape[1] + 1)
            input_ = self.__generate_sample(input_dims)
            build_dict['input_shape'] = input_dims
            original_input = input_.copy()

            weight = np.random.normal(-10., 10., size = (build_dict['kernel_size'],
                                            input_dims[-1], build_dict['filters'])).astype(np.float32)
            bias = np.random.normal(-10., 10., size = (build_dict['filters'])).astype(np.float32)

            weight_ptr = list_2_swig_float_pointer(weight.flatten().tolist(), weight.size)
            bias_ptr = list_2_swig_float_pointer(bias.flatten().tolist(), bias.size)

            c_output, output_dims = self.__c_fwd(build_dict, input_,
                                                 weight_ptr, bias_ptr, weight.size,
                                                 bias.size, input_dims)

            c_keras = self.__keras_fwd(build_dict, original_input, weight, bias)
            c_output = np.array(c_output).reshape(c_keras.shape)
            np.testing.assert_allclose(c_output, c_keras, atol = 1e-3, rtol = 1e-3)

if __name__=='__main__':
    unittest.main()