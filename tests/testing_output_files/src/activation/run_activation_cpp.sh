swig -python -c++ activation.i
c++ -c -fpic ../../code_test/activations.cpp
c++ -c -fpic activation_wrap.cxx -I/usr/include/python3.8
c++ -shared activations.o activation_wrap.o -o _activation.so
