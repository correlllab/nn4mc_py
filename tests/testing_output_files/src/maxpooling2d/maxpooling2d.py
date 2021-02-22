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
    from . import _maxpooling2d
else:
    import _maxpooling2d

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



def build_layer_maxpooling2d(pool_size_0, pool_size_1, strides_0, strides_1, padding, input_shape_0, input_shape_1, input_shape_2):
    return _maxpooling2d.build_layer_maxpooling2d(pool_size_0, pool_size_1, strides_0, strides_1, padding, input_shape_0, input_shape_1, input_shape_2)

def fwd_maxpooling2d(L, input):
    return _maxpooling2d.fwd_maxpooling2d(L, input)
class MaxPooling2D(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    pool_size = property(_maxpooling2d.MaxPooling2D_pool_size_get, _maxpooling2d.MaxPooling2D_pool_size_set)
    strides = property(_maxpooling2d.MaxPooling2D_strides_get, _maxpooling2d.MaxPooling2D_strides_set)
    padding = property(_maxpooling2d.MaxPooling2D_padding_get, _maxpooling2d.MaxPooling2D_padding_set)
    input_shape = property(_maxpooling2d.MaxPooling2D_input_shape_get, _maxpooling2d.MaxPooling2D_input_shape_set)
    output_shape = property(_maxpooling2d.MaxPooling2D_output_shape_get, _maxpooling2d.MaxPooling2D_output_shape_set)

    def __init__(self):
        _maxpooling2d.MaxPooling2D_swiginit(self, _maxpooling2d.new_MaxPooling2D())
    __swig_destroy__ = _maxpooling2d.delete_MaxPooling2D

# Register MaxPooling2D in _maxpooling2d:
_maxpooling2d.MaxPooling2D_swigregister(MaxPooling2D)

class input(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, nelements):
        _maxpooling2d.input_swiginit(self, _maxpooling2d.new_input(nelements))
    __swig_destroy__ = _maxpooling2d.delete_input

    def __getitem__(self, index):
        return _maxpooling2d.input___getitem__(self, index)

    def __setitem__(self, index, value):
        return _maxpooling2d.input___setitem__(self, index, value)

    def cast(self):
        return _maxpooling2d.input_cast(self)

    @staticmethod
    def frompointer(t):
        return _maxpooling2d.input_frompointer(t)

# Register input in _maxpooling2d:
_maxpooling2d.input_swigregister(input)

def input_frompointer(t):
    return _maxpooling2d.input_frompointer(t)



