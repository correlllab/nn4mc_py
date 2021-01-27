import conv1d
import tensorflow as tf
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
        new_object += [y[i]]
    return new_object

def list_2_swig_float_pointer(list : List[float], size : int):
    """
        Converts from list of floats to swig float* object
    """
    test_buffer = conv1d.input(size)
    for i in range(size):
        test_buffer[i] = list[i]
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
        print("generate_sample")
        return np.random.normal(0.0, 20, size = input_dims)

    def __keras_build(self, build_dict : dict):
        print("__keras_build")
        return tf.keras.layers.Conv1D(
                    filters = build_dict['filters'],
                    kernel_size = build_dict['kernel_size'],
                    strides = build_dict['strides'],
                    padding = build_dict['padding'],
                    data_format = build_dict['data_format'],
                    dilation_rate =  build_dict['dilation_rate'],
                    activation = build_dict['activation'],
                    use_bias = build_dict['use_bias']
                    )

    def test_padding(self):
        print("__test_padding")
        shape = np.random.randint(3, size=2).tolist()
        input_dims = (1, shape[0] + 1, shape[1] + 1)
        print(input_dims)
        input_ = self.__generate_sample(input_dims)
        build_dict = {'filters': 32, 'kernel_size': 3, 'strides': 1, 'padding': 'valid',
                      'data_format': 'channels_last', 'dilation_rate': 1, 'activation': 'linear',
                      'use_bias': True}

        weight : Final = conv1d.input(shape[0]*shape[1])
        bias : Final = conv1d.input(shape[1])

        padding = [0x00, 0x02, 0x03]
        print(input_.flatten().tolist())
        input_all = list_2_swig_float_pointer(input_.flatten().tolist(), input_dims[1]*input_dims[2])

        for pad in padding:
            input = copy.copy(input_all)
            print("padding: ", pad)
            layer = conv1d.build_layer_conv1d(weight.cast(), bias.cast(),
                                              build_dict['kernel_size'], build_dict['strides'],
                                              input_.shape[1], input_.shape[2], build_dict['filters'],
                                              activation_dictionary[build_dict['activation']],
                                              pad,
                                              dataformat_dictionary[build_dict['data_format']],
                                              build_dict['dilation_rate'])
            if (pad == 0x00):
                new_size =  len(input_.flatten().tolist())
            if (pad == 0x02):
                left_pad = build_dict['dilation_rate'] * (build_dict['kernel_size'] - 1)
                new_size = len(input_.flatten().tolist()) + input_dims[1]*left_pad
            if (pad == 0x03):
                left_pad = build_dict['filters'] // 2
                new_size = len(input_.flatten().tolist()) + input_dims[1]*left_pad

            padding_result = conv1d.padding_1d(layer, input.cast())

            padding_result = swig_py_object_2_list(padding_result, new_size)



    #def __c_fwd(self, build_dict : dict, input_, weight, bias):
    #    print("__c_fwd")
    #    input_list = input_.flatten().tolist()

    #    weight_ptr = (ctypes.c_float).from_address(int(weight))
    #    bias_ptr = (ctypes.c_float).from_address(int(bias))

    #    iinput = []
    #    for i in range(len(input_list)):
    #        iinput += [input_list[i]]

    #    layer = conv1d.build_layer_conv1d(weight_ptr, bias_ptr,
    #                    build_dict['kernel_size'], build_dict['strides'],
    #                    input_.shape[1], input_.shape[2], build_dict['filters'],
    #                    activation_dictionary[build_dict['activation']],
    #                    padding_dictionary[build_dict['padding']],
    #                    dataformat_dictionary[build_dict['data_format']],
    #                    build_dict['dilation_rate'])
    #    return conv1d.fwd_conv1d(layer, iinput.cast())

    #def __keras_fwd(self, config_dict : dict, input_, weight, bias):
    #    print("__keras_fwd")
    #    layer = self.__keras_build(config_dict)
    #    layer.w = tf.Variable(weight)
    #    layer.b = tf.Variable(bias)
    #    return layer(input_)

    #def test_fwd(self) -> bool:
    #    print("test_fwd")
    #    batch_size = 1
    #    N = 1000
    #    for _ in range(N):
    #        build_dict = {'filters': 32, 'kernel_size' : 3, 'strides' : 1, 'padding' : 'valid',
    #                'data_format' : 'channels_last', 'dilation_rate' : 1, 'activation' : 'linear',
    #                'use_bias' : True}

    #        shape = np.random.randint(100, size = 2).tolist()

    #        input_dims = (batch_size, shape[0] , shape[1])

    #        weight = np.random.normal(0.0, 20., size = (build_dict['kernel_size'],
    #                                        input_dims[-1], build_dict['filters'])).astype(np.float32)
    #        bias = np.random.normal(0.0, 20, size = (build_dict['filters'])).astype(np.float32)

    #        input_ = self.generate_sample(input_dims)

    #        weight_ptr = (ctypes.c_float).from_address(int(weight[0][0][0]))
    #        bias_ptr = (ctypes.c_float).from_address(int(bias[0]))

    #        self.assertEqual(self.__c_fwd(build_dict, input_, weight_ptr,
    #                    bias_ptr), self.__keras_fwd(build_dict, input_, weight, bias))

if __name__=='__main__':
    unittest.main()