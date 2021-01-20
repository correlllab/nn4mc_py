
/********************
    activations.cpp

    Code generated using nn4mc.

    This file implements all activation functions.

*/

#include "activations.h"
#include <math.h>
#include <stdlib.h>

#define max(a, b) (((a)>(b) ? (a) : (b)))
#define min(a, b) (((a)<(b) ? (a) : (b)))

float * sigmoid(float * input, int m)
{
    for (int i = m - 1; i>= 0; i--){
        if (input[i] >= 0.0){
            input[i] = 1./(exponential(-input[i]) + 1.);
        } else{
            input[i] = exponential(input[i]) / (1. + exponential(input[i]));
        }

       if (isnan(input[i])){
            input[i] = 1.;
       }
   }
  return input;
}

float * softplus(float * input, int m)
{
  for (int i = m - 1; i>= 0; i--){
      float x = input[i];
      input[i] = log(exponential(input[i]) + 1.);

      if (isinf(input[i])){
        input[i] = x;
      }
    }
  return input;
}

float* softsign(float *input, int m)
{
   for (int i = m-1; i>=0; i--){
        input[i] = input[i] / (abs(input[i]) + 1);
    }
  return input;
}

float * hard_sigmoid(float * input, int m)
{
    for (int i = m - 1; i >= 0; i--){
         input[i] = input[i] * 0.2 + 0.5;
          if (input[i] < 0){
              input[i] = 0.0;
          } else if (input[i] > 1.0){
              input[i] = 1.0;
          }
      }

  return input;
}

float exponential(float input)
{
// not an activation function
  input = (float)expf((float)input);
  return input;
}

float * relu(float * input, int m)
{
  for (int i = m - 1; i>=0; i--){
        input[i] = max(input[i], 0.0);
  }
  return input;
}

float * exp_activation(float * input, int m)
{
  for (int i = m - 1; i>=0; i--){
        input[i] = exponential(input[i]);
  }
  return input;
}


float * hyper_tan(float* input, int m)
{
  for (int i = m - 1; i>= 0; i--){
        input[i] = tanh(input[i]);
  }
  return input;
}

float * softmax(float * input, int m)
{
  float sum_exp = 0.0;
  for (int i=0; i<m; i++){
      sum_exp+= exponential(input[i]);
  }
  for (int i=0; i<m;i++){
      float calc = exponential(input[i]) / sum_exp;
      if (isnan(calc)){
          input[i] = 1.0;
      } else input[i] = (float)(exponential(input[i]) / sum_exp);
  }
  return input;
}

float* activate(float * input, int output_shape, char type)
{
  if (type == 0x00)
    return softmax(input, output_shape);

  /* else if (type == 0x02)
    return elu(); */

  /* else if (type == 0x03)
    return selu(); */

  else if (type == 0x04)
    return softplus(input, output_shape);

  else if (type == 0x05)
    return softsign(input, output_shape);

  else if (type == 0x06)
    return relu(input, output_shape);

  else if (type == 0x07)
    return hyper_tan(input, output_shape);

  else if (type == 0x08)
    return sigmoid(input, output_shape);

  else if (type == 0x09)
    return hard_sigmoid(input, output_shape);

  else if (type == 0xA)
    return exp_activation(input, output_shape);

  return input;
}

