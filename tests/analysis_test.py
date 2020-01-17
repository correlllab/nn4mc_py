#!/usr/bin/env python
import sys
import nn4mp_py

def analyze():
    AL = nn4mp.Analyzer()

    AL.load_model(sys.argv[0])
    AL.cheapify()
    AL.compare()


if __name__ == '__main__':
    analyze()
