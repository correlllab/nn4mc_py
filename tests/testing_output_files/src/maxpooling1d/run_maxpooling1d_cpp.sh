swig -python -c++ maxpooling1d.i
c++ -c -fpic  ../../code_test/maxpooling1d.cpp
c++ -c -fpic maxpooling1d_wrap.cxx -I/usr/include/python3.8
c++ -shared maxpooling1d.o maxpooling1d_wrap.o -o _maxpooling1d.so
