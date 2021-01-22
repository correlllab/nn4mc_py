swig -python -c++ conv1d.i
c++ -c -fpic ../../code_test/activations.cpp ../../code_test/conv1d.cpp
c++ -c -fpic conv1d_wrap.cxx -I/usr/include/python3.8
c++ -shared conv1d.o conv1d_wrap.o -o _conv1d.so
