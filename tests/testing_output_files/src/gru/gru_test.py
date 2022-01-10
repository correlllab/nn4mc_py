import gru
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Input
from tensorflow.keras.backend import clear_session
import numpy as np
import unittest
from typing import List, Final
import ctypes
import copy
import matplotlib.pyplot as plt

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
    test_buffer = gru.input(size)
    for i in range(size):
        test_buffer[i] = float(list[i])
    return test_buffer


activation_dictionary = {'softmax': 0x00,
                         'elu': 0x02, 'selu': 0x03,
                         'softplus': 0x04, 'softsign': 0x05,
                         'relu': 0x06, 'tanh': 0x07, 'sigmoid': 0x08,
                         'hard_sigmoid': 0x09, 'exponential': 0xA, 'linear': 0xB,
                         'custom': 0xC}

class GRUTest(unittest.TestCase):
    """
        GRU Testing Module
    """
    def __generate_sample(self, input_dims):
        return np.random.normal(-0.5, 0.5, size = input_dims).astype(np.float64)

    def __keras_build(self, build_dict : dict):
        model = Sequential()
        model.add(GRU(
                    input_shape = build_dict['input_shape'],
                    #activation = build_dict['activation'],
                    units = build_dict['units'],
                    bias_initializer='zeros',
                    trainable = False,
                    #recurrent_activation= build_dict['recurrent_activation'],
                    use_bias = build_dict['use_bias'],
                    ))
        return model

    def __c_fwd(self, build_dict : dict, input_, weight, big_u,
                                    bias, weight_size, big_u_size,
                                    bias_size, input_dims, units):

        weight = list_2_swig_float_pointer(weight, weight_size)
        big_u = list_2_swig_float_pointer(big_u, big_u_size)
        bias = list_2_swig_float_pointer(bias, bias_size)

        input_length = input_.size
        input_all = list_2_swig_float_pointer(input_.flatten().tolist(), input_length)
        output_dims = units * input_dims[0]

        layer = gru.build_layer_gru(weight.cast(), big_u.cast(), bias.cast(),
                                    activation_dictionary[build_dict['recurrent_activation']],
                                    activation_dictionary[build_dict['activation']],
                                    input_dims[0], input_dims[1], units)

        output = gru.fwd_gru(layer, input_all.cast())
        output = swig_py_object_2_list(output, output_dims)
        return output, output_dims

    def __keras_fwd(self, config_dict : dict, input_, weight, big_u, bias):
        model = self.__keras_build(config_dict)
        model.set_weights([weight, big_u, bias])
        prediction = model.predict(input_)
        del model
        clear_session()
        return prediction

    def test_fwd(self):
        N = 1000
        for _ in range(N):
            units = int(np.random.randint(1, 3, size = 1)[0])
            build_dict = {'activation' : 'tanh',
                          'recurrent_activation' : 'hard_sigmoid',
                          'units': units,
                          'use_bias' : True, 'input_shape': None}
            shape = np.random.randint(25, 100, size = 2).tolist()
            shape[0] = 1
            input_ = self.__generate_sample(shape)
            build_dict['input_shape'] = shape

            weight = np.random.normal(-1e-3, 1e-3, size = (shape[-1], build_dict['units']*3)).astype(np.float64)
            big_u = np.random.normal(-1e-3, 1e-3, size = (build_dict['units'], build_dict['units']*3)).astype(np.float64)
            bias = np.random.normal(-1e-3, 1e-3, size = (2, build_dict['units']*3)).astype(np.float64)

            weight_ptr = list_2_swig_float_pointer(weight.flatten().tolist(), weight.size)
            big_u_ptr = list_2_swig_float_pointer(big_u.flatten().tolist(), big_u.size)
            bias_ptr = list_2_swig_float_pointer(bias.flatten().tolist(), bias.size)

            c_output, output_dims = self.__c_fwd(build_dict, input_,
                                                 weight_ptr, big_u_ptr,
                                                 bias_ptr, weight.size,
                                                 big_u.size, bias.size,
                                                 shape, units)

            output_keras = self.__keras_fwd(build_dict,
                                            input_.reshape(1, shape[0], shape[1]),
                                            weight, big_u, bias)
            #product = np.dot(input_, weight[:, units:2*units]) + bias[0, units:2*units]
            #product += bias[1, units:2*units]

            #np.testing.assert_allclose(np.clip(product * 0.2 + 0.5, 0, 1), np.array(c_output).reshape(product.shape), rtol = 1e-5)
            output_c = np.array(c_output).reshape(output_keras.shape).astype(np.float32)
            print(build_dict)
            print("c:", output_c.reshape(output_keras.shape))
            print("keras:", output_keras)
            print("error: ", abs(output_c.reshape(output_keras.shape) - output_keras))
            np.testing.assert_allclose(output_c, output_keras,
                                      rtol = 7e-3, atol = 7e-3)

if __name__=='__main__':
    unittest.main()