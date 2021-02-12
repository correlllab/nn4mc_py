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
    #include "../../code_test/conv2d.h"
    #include "../../code_test/parameters.h"
%}
%include "../../code_test/conv2d.cpp"
%include "../../code_test/conv2d.h"
%include "../../code_test/parameters.h"
%include "../../code_test/activations.h"
%include "../../code_test/activations.cpp"
%include "stdint.i"
%include "carrays.i"

%module conv2d

%array_class(float, input);

%{
    extern struct Conv2D build_layer_conv2d(const float*, const float*, int, int, int, int, int, int, int, int, char,char,char, int , int);
   
    extern float* fwd_conv2d(struct Conv2D, float*);

    extern float * padding_2d(struct Conv2D, float *, int*, int*);

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

extern struct Conv2D build_layer_conv2d(const float*, const float*, int, int, int, int, int, int, int, int, char,char,char, int , int);

extern float* fwd_conv2d(struct Conv2D, float*);

extern float * padding_2d(struct Conv2D, float *, int*, int*);

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

