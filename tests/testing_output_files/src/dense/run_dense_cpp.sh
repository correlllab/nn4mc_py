swig -python -c++ dense.i
c++ -c -fpic ../../code_test/activations.cpp ../../code_test/dense.cpp
c++ -c -fpic dense_wrap.cxx -I/usr/include/python3.8
c++ -shared dense.o activations.o dense_wrap.o -o _dense.so
