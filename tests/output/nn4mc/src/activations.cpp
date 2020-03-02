
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


float relu(float input)
{
  input = max(input, 0.0);

  return input;
}



//NOTE: This is deprecated and need to be changed to be more streamlined
float activate(float input, int output_shape, char type)
{
  if (type == 0x00)
    return softmax(input, output_shape);

  /* else if (type == 0x02)
    return elu(); */

  /* else if (type == 0x03)
    return selu(); */

  else if (type == 0x04)
    return softplus(input);

  else if (type == 0x05)
    return softsign(input);

  else if (type == 0x06)
    return relu(input);

  else if (type == 0x07)
    return hyper_tan(input);

  else if (type == 0x08)
    return sigmoid(input);

  else if (type == 0x09)
    return hard_sigmoid(input);

  else if (type == 0xA)
    return exponential(input);

  /* else if (type == 0xC)
    return custom(input); */

  return input;
}

