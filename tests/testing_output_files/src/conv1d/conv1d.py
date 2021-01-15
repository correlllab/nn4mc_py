# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.1
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _conv1d
else:
    import _conv1d

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)



def buildConv1D(W, b, kernel_size, strides, input_sh0, input_sh1, filters, activation, padding, data_format, dilation_rate):
    return _conv1d.buildConv1D(W, b, kernel_size, strides, input_sh0, input_sh1, filters, activation, padding, data_format, dilation_rate)

def padding_conv1(L, input):
    return _conv1d.padding_conv1(L, input)

def fwdConv1D(L, input):
    return _conv1d.fwdConv1D(L, input)
class Conv1D(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    weights = property(_conv1d.Conv1D_weights_get, _conv1d.Conv1D_weights_set)
    biases = property(_conv1d.Conv1D_biases_get, _conv1d.Conv1D_biases_set)
    strides = property(_conv1d.Conv1D_strides_get, _conv1d.Conv1D_strides_set)
    kernel_shape = property(_conv1d.Conv1D_kernel_shape_get, _conv1d.Conv1D_kernel_shape_set)
    weight_shape = property(_conv1d.Conv1D_weight_shape_get, _conv1d.Conv1D_weight_shape_set)
    filters = property(_conv1d.Conv1D_filters_get, _conv1d.Conv1D_filters_set)
    dilation_rate = property(_conv1d.Conv1D_dilation_rate_get, _conv1d.Conv1D_dilation_rate_set)
    activation = property(_conv1d.Conv1D_activation_get, _conv1d.Conv1D_activation_set)
    padding = property(_conv1d.Conv1D_padding_get, _conv1d.Conv1D_padding_set)
    data_format = property(_conv1d.Conv1D_data_format_get, _conv1d.Conv1D_data_format_set)
    input_shape = property(_conv1d.Conv1D_input_shape_get, _conv1d.Conv1D_input_shape_set)
    output_shape = property(_conv1d.Conv1D_output_shape_get, _conv1d.Conv1D_output_shape_set)

    def __init__(self):
        _conv1d.Conv1D_swiginit(self, _conv1d.new_Conv1D())
    __swig_destroy__ = _conv1d.delete_Conv1D

# Register Conv1D in _conv1d:
_conv1d.Conv1D_swigregister(Conv1D)


def activate(arg1, arg2, arg3):
    return _conv1d.activate(arg1, arg2, arg3)

def sigmoid(arg1):
    return _conv1d.sigmoid(arg1)

def softplus(arg1):
    return _conv1d.softplus(arg1)

def softsign(input):
    return _conv1d.softsign(input)

def hard_sigmoid(input):
    return _conv1d.hard_sigmoid(input)

def exponential(input):
    return _conv1d.exponential(input)

def relu(input):
    return _conv1d.relu(input)

def hyper_tan(input):
    return _conv1d.hyper_tan(input)

def softmax(input, output_shape):
    return _conv1d.softmax(input, output_shape)

cvar = _conv1d.cvar
conv1d_1_W = cvar.conv1d_1_W
conv1d_1_b = cvar.conv1d_1_b
conv1d_2_W = cvar.conv1d_2_W
conv1d_2_b = cvar.conv1d_2_b
dense_1_W = cvar.dense_1_W
dense_1_b = cvar.dense_1_b
dense_2_W = cvar.dense_2_W
dense_2_b = cvar.dense_2_b
dense_3_W = cvar.dense_3_W
dense_3_b = cvar.dense_3_b
dense_4_W = cvar.dense_4_W
dense_4_b = cvar.dense_4_b
dense_5_W = cvar.dense_5_W
dense_5_b = cvar.dense_5_b
dense_6_W = cvar.dense_6_W
dense_6_b = cvar.dense_6_b
dense_7_W = cvar.dense_7_W
dense_7_b = cvar.dense_7_b
dense_8_W = cvar.dense_8_W
dense_8_b = cvar.dense_8_b

