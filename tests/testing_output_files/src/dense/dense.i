/* File: dense.i:

    This is the file that gives us enough to wrap the code. 
    Here we introduce headers that import the functions or header
    files. Since activations.h also imports all of the function 
    prototypes from activation.cpp, you only need to import the 
    header.
*/
%{
    #define SWIG_FILE_WITH_INIT
    #include "../../code_test/dense.h"
    #include "../../code_test/parameters.h"
%}
%include "../../code_test/dense.cpp"
%include "../../code_test/dense.h"
%include "../../code_test/parameters.h"
%include "stdint.i"
%include "carrays.i"

%module dense

%array_class(float, input);

%{
    struct Dense build_layer_dense(const float*, const float*, int, int, char);

    float * fwd_dense(struct Dense, float* );

    extern float * activate(float* input, int output_shape, char type);

    extern float * sigmoid(float * input, int m);

    extern float * exp_activation(float * input, int m);

    extern float * softplus(float * input, int m);

    extern float * softsign(float * input, int m);

    extern float * hard_sigmoid(float * input, int m);

    extern float  exponential(float input);

    extern float * relu(float *input, int m);

    extern float * hyper_tan(float * input, int m);

    extern float * softmax(float * input, int m);
%}

struct Dense build_layer_dense(const float*, const float*, int, int, char);

float * fwd_dense(struct Dense, float*);

extern float * activate(float* input, int output_shape, char type);

extern float * sigmoid(float * input, int m);

extern float * exp_activation(float * input, int m);

extern float * softplus(float * input, int m);

extern float * softsign(float * input, int m);

extern float * hard_sigmoid(float * input, int m);

extern float  exponential(float input);

extern float * relu(float *input, int m);

extern float * hyper_tan(float * input, int m);

extern float * softmax(float * input, int m);

