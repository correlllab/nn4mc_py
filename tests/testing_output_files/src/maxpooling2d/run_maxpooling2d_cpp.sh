swig -python -c++ maxpooling2d.i
c++ -c -fpic  ../../code_test/maxpooling2d.cpp
c++ -c -fpic maxpooling2d_wrap.cxx -I/usr/include/python3.8
c++ -shared maxpooling2d.o maxpooling2d_wrap.o -o _maxpooling2d.so
