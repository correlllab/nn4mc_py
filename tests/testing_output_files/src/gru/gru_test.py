import gru
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU
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
        return np.random.normal(0.0, 20, size = input_dims)

    def __keras_build(self, build_dict : dict):
        model = Sequential()
        model.add(GRU(
                    input_shape = build_dict['input_shape'][1:],
                    activation = build_dict['activation'],
                    units = build_dict['units'],
                    recurrent_activation= build_dict['recurrent_activation'],
                    use_bias = build_dict['use_bias']
                    ))
        model.trainable = False
        return model

    def __c_fwd(self, build_dict : dict, input_, weight, big_u,
                                    bias, weight_size, bias_size, input_dims, units):
        weight = list_2_swig_float_pointer(weight, weight_size)
        big_u = list_2_swig_float_pointer(big_u, weight_size)
        bias = list_2_swig_float_pointer(bias, bias_size)
        input_length = input_.size

        input_ = input_.flatten().tolist()
        input_all = list_2_swig_float_pointer(input_, len(input_))

        output_dims = units
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
            units = int(np.random.randint(1, 5, size = 1)[0])
            build_dict = {'activation' : 'tanh',
                          'recurrent_activation' : 'sigmoid',
                          'units': units,
                          'use_bias' : True, 'input_shape': None}
            shape = np.random.randint(1, 5, size = 2).tolist()
            input_dims = (1, shape[0], shape[1])
            input_ = self.__generate_sample(input_dims)
            build_dict['input_shape'] = input_dims
            original_input = input_.copy()
            weight = np.random.normal(-10., 10., size = (input_dims[-1], build_dict['units']*3)).astype(np.float32)
            big_u = np.random.normal(-10., 10., size = (build_dict['units'], build_dict['units']*3)).astype(np.float32)
            bias = np.random.normal(-10., 10., size = (2, build_dict['units']*3)).astype(np.float32)
            weight_ptr = list_2_swig_float_pointer(weight.flatten().tolist(), weight.size)
            big_u_ptr = list_2_swig_float_pointer(big_u.flatten().tolist(), big_u.size)
            bias_ptr = list_2_swig_float_pointer(bias.flatten().tolist(), bias.size)
            c_output, output_dims = self.__c_fwd(build_dict, input_,
                                                 weight_ptr, big_u_ptr, bias_ptr, weight.size,
                                                 bias.size, input_dims, units)
            output_keras = self.__keras_fwd(build_dict, original_input, weight, big_u, bias)
            output_c = np.array(c_output).reshape(output_keras.shape)
            np.testing.assert_allclose(output_c, output_keras, atol = 1e-3, rtol = 1e-3)

if __name__=='__main__':
    unittest.main()