
/********************
    nn4mc.h

    Code generated using nn4mc.

    This file defines a a neural network and associated functions.

*/

#ifndef __NEURAL_NETWORK_H__
#define __NEURAL_NETWORK_H__

#ifdef __cplusplus
extern "C" {

#include "parameters.h"
#include gru.h
#include dense.h


gru gru;
dense dense;


float* fwdNN(float*);
void buildLayers();

}
#endif
#endif

