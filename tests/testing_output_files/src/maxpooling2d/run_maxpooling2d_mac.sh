swig -python -c++ maxpooling2d.i
c++ -c -fpic  ../../code_test/maxpooling2d.cpp
c++ -c -fpic maxpooling2d_wrap.cxx -I/usr/local/Cellar/python@3.8/3.8.5/Frameworks/Python.framework/Versions/3.8/include/python3.8
c++ -bundle -flat_namespace maxpooling2d.o maxpooling2d_wrap.o -undefined suppress -o _maxpooling2d.so

