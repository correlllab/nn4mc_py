/* File: activation.i:

    This is the file that gives us enough to wrap the code. 
    Here we introduce headers that import the functions or header
    files. Since activations.h also imports all of the function 
    prototypes from activation.cpp, you only need to import the 
    header.
*/
%{
    #define SWIG_FILE_WITH_INIT
    #include "../../code_test/activations.h"
    #include "../../code_test/gru.h"
    #include "../../code_test/parameters.h"
%}
%include "../../code_test/gru.cpp"
%include "../../code_test/gru.h"
%include "../../code_test/parameters.h"
%include "../../code_test/activations.h"
%include "../../code_test/activations.cpp"
%include "stdint.i"
%include "carrays.i"

%module gru

%array_class(float, input);

%{
    extern struct GRU build_layer_gru(const float*, const float*, const float*,
                            char, char, int, int, int);

    extern float * fwd_gru(struct GRU, float *);

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

extern struct  GRU build_layer_gru(const float*, const float*, const float*,
                            char, char, int, int, int);

extern float * fwd_gru(struct GRU, float *);

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

