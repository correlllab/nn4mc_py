swig -python -c++ conv2d.i
c++ -c -fpic ../../code_test/activations.cpp ../../code_test/conv2d.cpp
c++ -c -fpic conv2d_wrap.cxx -I/usr/include/python3.8
c++ -shared conv2d.o activations.o conv2d_wrap.o -o _conv2d.so
