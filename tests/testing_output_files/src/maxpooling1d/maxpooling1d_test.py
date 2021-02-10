import maxpooling1d
from tensorflow.keras import Sequential
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.backend import clear_session
import numpy as np
import unittest
from typing import List, Final
import ctypes
import copy

padding_dictionary = {'valid': 0x00, 'causal': 0x02, 'same': 0x03}
dataformat_dictionary = {'channels_last': 0x00, 'channels_first': 0x02}

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
    test_buffer = maxpooling1d.input(size)
    for i in range(size):
        test_buffer[i] = float(list[i])
    return test_buffer

class MaxPooling1DTest(unittest.TestCase):
    """
        MaxPooling1D
    """
    def __generate_sample(self, input_dims):
        return np.random.normal(0.0, 20, size = input_dims)

    def __keras_build(self, build_dict : dict):
        model = Sequential()
        model.add(MaxPooling1D(
                    pool_size = build_dict['pool_size'],
                    strides = build_dict['strides'],
                    padding = build_dict['padding'],
                    data_format = build_dict['data_format'],
                    ))
        model.trainable = False
        return model

    def __c_fwd(self, build_dict : dict, input_, input_dims, output_dims):

        input_ = input_.flatten().tolist()
        input_all = list_2_swig_float_pointer(input_, len(input_))
        layer = maxpooling1d.build_layer_maxpooling1d(build_dict['pool_size'],
                                                      build_dict['strides'],
                                                      input_dims[1],
                                                      input_dims[2],
                                                      padding_dictionary[build_dict['padding']])

        output = maxpooling1d.fwd_maxpooling1d(layer, input_all.cast())
        print("keras output", output_dims)
        c_output_size = int(np.ceil((input_dims[1] - build_dict['pool_size'] + 1) / build_dict['strides']) * input_dims[2])
        print("c_output size", (1,int(np.ceil((input_dims[1] - build_dict['pool_size'] + 1) / build_dict['strides'])) ,input_dims[2]))
        output = swig_py_object_2_list(output, c_output_size)

        return output, output_dims

    def __keras_fwd(self, config_dict : dict, input_):
        model = self.__keras_build(config_dict)
        prediction = model.predict(input_)
        del model
        clear_session()
        return prediction

    def test_fwd(self):
        N = 1000
        for _ in range(N):
            pool_size = np.random.randint(1, 10, size=1).tolist()[0]
            strides = np.random.randint(1, 10, size=1).tolist()[0]

            build_dict = {'pool_size' : pool_size ,
                          'strides' :  strides,
                          'padding' : 'valid',
                          'data_format' : "channels_last"}

            shape = np.random.randint(50, 100, size = 2).tolist()
            input_dims = (1, shape[0], shape[1])
            print("input: ", input_dims)
            input_ = self.__generate_sample(input_dims)
            build_dict['input_shape'] = input_dims
            original_input = input_.copy()

            c_keras = np.array(self.__keras_fwd(build_dict, original_input))
            c_output, output_dims = self.__c_fwd(build_dict, input_, input_dims, c_keras.shape)

            c_output = np.array(c_output).reshape(c_keras.shape).astype(np.float32)
            print(c_keras)
            print(c_output)
            assert_result = np.testing.assert_allclose(c_output.flatten(), c_keras.flatten(), rtol = 1e-5)
            print(assert_result)

        print("forward passed!")

if __name__=='__main__':
    unittest.main()