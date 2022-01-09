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
    from . import _activation
else:
    import _activation

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



def activate(arg1, arg2, arg3):
    return _activation.activate(arg1, arg2, arg3)

def sigmoid(arg1, arg2):
    return _activation.sigmoid(arg1, arg2)

def softplus(arg1, arg2):
    return _activation.softplus(arg1, arg2)

def softsign(arg1, arg2):
    return _activation.softsign(arg1, arg2)

def hard_sigmoid(arg1, arg2):
    return _activation.hard_sigmoid(arg1, arg2)

def exp_activation(arg1, arg2):
    return _activation.exp_activation(arg1, arg2)

def exponential(arg1):
    return _activation.exponential(arg1)

def relu(arg1, arg2):
    return _activation.relu(arg1, arg2)

def elu(arg1, arg2, arg3):
    return _activation.elu(arg1, arg2, arg3)

def selu(arg1, arg2):
    return _activation.selu(arg1, arg2)

def hyper_tan(arg1, arg2):
    return _activation.hyper_tan(arg1, arg2)

def softmax(arg1, arg2):
    return _activation.softmax(arg1, arg2)
class input(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, nelements):
        _activation.input_swiginit(self, _activation.new_input(nelements))
    __swig_destroy__ = _activation.delete_input

    def __getitem__(self, index):
        return _activation.input___getitem__(self, index)

    def __setitem__(self, index, value):
        return _activation.input___setitem__(self, index, value)

    def cast(self):
        return _activation.input_cast(self)

    @staticmethod
    def frompointer(t):
        return _activation.input_frompointer(t)

# Register input in _activation:
_activation.input_swigregister(input)

def input_frompointer(t):
    return _activation.input_frompointer(t)



