/* File: maxpooling1d.i:

    This is the file that gives us enough to wrap the code. 
    Here we introduce headers that import the functions or header
    files. Since activations.h also imports all of the function 
    prototypes from activation.cpp, you only need to import the 
    header.
*/
%{
    #define SWIG_FILE_WITH_INIT
    #include "../../code_test/maxpooling1d.h"
%}
%include "../../code_test/maxpooling1d.cpp"
%include "../../code_test/maxpooling1d.h"
%include "stdint.i"
%include "carrays.i"

%module maxpooling1d

%array_class(float, input);

%{
    struct MaxPooling1D build_layer_maxpooling1d(int, int, int, int, char);

    float * fwd_maxpooling1d(struct MaxPooling1D, float*);
%}

struct MaxPooling1D build_layer_maxpooling1d(int, int, int, int, char);

float * fwd_maxpooling1d(struct MaxPooling1D, float*);

