/* File: maxpooling2d.i:

    This is the file that gives us enough to wrap the code. 
    Here we introduce headers that import the functions or header
    files. Since activations.h also imports all of the function 
    prototypes from activation.cpp, you only need to import the 
    header.
*/
%{
    #define SWIG_FILE_WITH_INIT
    #include "../../code_test/maxpooling2d.h"
%}
%include "../../code_test/maxpooling2d.cpp"
%include "../../code_test/maxpooling2d.h"
%include "stdint.i"
%include "carrays.i"

%module maxpooling2d

%array_class(float, input);

%{
    struct MaxPooling2D build_layer_maxpooling2d(int, int, int, int, char, int, int, int);

    float * fwd_maxpooling2d(struct MaxPooling2D , float *);
%}

struct MaxPooling2D build_layer_maxpooling2d(int, int, int, int, char, int, int, int);

float * fwd_maxpooling2d(struct MaxPooling2D , float *);

