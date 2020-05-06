# Neural Networks for Microcontrollers: Python

This is a python implementation of Neural Networks for Microcontrollers (nn4mc) (originally implemented in C++) that allows for the translation of trained neural network models to C code for use in embedded systems.

## Development Status

Please note that nn4mc_py is still in development and may have many bugs. We are working hard on getting everything operating seamlessly, and feel free to reach out with any questions.

## Using nn4mc

### Installation

Simply use the Python package manager pip and run the following command.
'''
pip install nn4mc
'''

### Usage

You will most likely only need to import the translator module
'''python
import nn4mc.translator as nnTr
'''

Then you can translate a file with the following command
'''python
nnTr.translate(path/to/file, 'hdf5', output/path)
'''
