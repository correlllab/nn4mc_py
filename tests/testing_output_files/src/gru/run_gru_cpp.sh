swig -python -c++ gru.i
c++ -c -fpic ../../code_test/activations.cpp ../../code_test/gru.cpp
c++ -c -fpic gru_wrap.cxx -I/usr/include/python3.8
c++ -shared gru.o activations.o gru_wrap.o -o _gru.so
