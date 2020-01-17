#NOTE: This file imports all of the submodules, but there
#probably a better way to do this.

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import nn4mc_py.analysis as nnAl
import nn4mc_py.datastructures as nnDs
import nn4mc_py.generator as nnGn
import nn4mc_py.parser as nnPr
import nn4mc_py.translator as nnTr
