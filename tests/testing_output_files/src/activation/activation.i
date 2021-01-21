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
%}

%include "../../code_test/activations.h"
%include "../../code_test/activations.cpp"
%include "stdint.i"
%include "carrays.i"

%module activation

%array_class(float, input);

%{

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

 extern float * activate(float * input, int output_shape, char type);
 
 extern float * sigmoid(float * input, int m);
 
 extern float * softplus(float * input, int m);

 extern float * exp_activation(float * input, int m);

 extern float * softsign(float * input, int m);
 
 extern float * hard_sigmoid(float * input, int m);
 
 extern float exponential(float input);
 
 extern float * relu(float* input, int m);

 extern float * hyper_tan(float * input, int m);
 
 extern float * softmax(float * input, int m);


