import dense
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
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
    test_buffer = dense.input(size)
    for i in range(size):
        test_buffer[i] = float(list[i])
    return test_buffer


activation_dictionary = {'softmax': 0x00,
                         'elu': 0x02, 'selu': 0x03,
                         'softplus': 0x04, 'softsign': 0x05,
                         'relu': 0x06, 'tanh': 0x07, 'sigmoid': 0x08,
                         'hard_sigmoid': 0x09, 'exponential': 0xA, 'linear': 0xB,
                         'custom': 0xC}

class Conv1DTest(unittest.TestCase):
    """
        Conv1D
    """
    def __generate_sample(self, input_dims):
        return np.random.normal(0.0, 20, size = input_dims)

    def __keras_build(self, build_dict : dict):
        model = Sequential()
        model.add(Dense(
                    input_shape = build_dict['input_shape'],
                    units = build_dict['units'],
                    ))
        model.trainable = False
        return model

    def __c_fwd(self, build_dict : dict, input_, units, weight, bias, weight_size, bias_size, input_dims):

        weight = list_2_swig_float_pointer(weight, weight_size)
        bias = list_2_swig_float_pointer(bias, bias_size)

        input_ = input_.flatten().tolist()
        input_all = list_2_swig_float_pointer(input_, len(input_))

        output_dims = units

        layer = dense.build_layer_dense(weight.cast(), bias.cast(),
                                              len(input_), output_dims,
                                              activation_dictionary[build_dict['activation']])

        output = dense.fwd_dense(layer, input_all.cast())
        output = swig_py_object_2_list(output, output_dims)
        return output, output_dims

    def __keras_fwd(self, config_dict : dict, input_, weight, bias):
        model = self.__keras_build(config_dict)
        model.set_weights([weight, bias])
        return model.predict(input_)

    def test_fwd(self):
        N = 1000
        assert_result = True
        for _ in range(N):
            units = np.random.randint(1, 10, size=1).tolist()[0]
            build_dict = {'activation' : 'linear', 'units' : units}

            shape = np.random.randint(1, 5, size = 1).tolist()[0]
            input_dims = (1, shape)
            input_ = self.__generate_sample(input_dims)
            build_dict['input_shape'] = input_dims
            original_input = input_.copy()

            weight = np.random.normal(-10., 10., size = (shape, units)).astype(np.float32)
            bias = np.random.normal(-10., 10., size = units).astype(np.float32)

            weight_ptr = list_2_swig_float_pointer(weight.flatten().tolist(), weight.size)
            bias_ptr = list_2_swig_float_pointer(bias.flatten().tolist(), bias.size)

            c_output, output_dims = self.__c_fwd(build_dict, input_, units,
                                                 weight_ptr, bias_ptr, weight.size,
                                                 bias.size, input_dims)

            c_keras = self.__keras_fwd(build_dict, original_input, weight, bias)
            c_output = np.array(c_output).reshape(c_keras.shape)
            assert_result = np.testing.assert_allclose(c_output, c_keras, rtol = 5e-5)
        return assert_result
if __name__=='__main__':
    unittest.main()