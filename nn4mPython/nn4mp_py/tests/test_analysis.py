from .context import nn4mp_py
import unittest

def analyze():
    AL = nn4mp.Analyzer()

    AL.load_model(sys.argv[0])
    AL.cheapify()
    AL.


if __name__ == '__main__':
    analyze()
