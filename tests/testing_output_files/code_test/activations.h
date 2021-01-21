
/********************
    activations.h

    Code generated using nn4mc.

    This file defines possible activation functions.

*/

#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

float* activate(float*, int, char);

float* sigmoid(float*, int);

float* softplus(float *, int);

float* softsign(float *, int);

float* hard_sigmoid(float *, int);

float* exp_activation(float *, int);

float exponential(float);

float* relu(float*, int);

float* elu(float*, int, float);

float* selu(float*, int);

float* hyper_tan(float *, int);

float* softmax(float *, int );

#endif
