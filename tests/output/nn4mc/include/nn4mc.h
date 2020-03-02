
/********************
    nn4mc.h

    Code generated using nn4mc.

    This file defines a a neural network and associated functions.

*/

#ifndef __NEURAL_NETWORK_H__
#define __NEURAL_NETWORK_H__

#ifdef __cplusplus
extern "C" {

#include "neural_network_params.h"
#include conv1d.h
#include dense.h


struct conv1d conv1d_1;
struct conv1d conv1d_2;
struct dense dense_1;
struct dense dense_2;
struct dense dense_3;
struct dense dense_4;
struct dense dense_5;
struct dense dense_6;
struct dense dense_7;
struct dense dense_8;


float* fwdNN(float*);
void buildLayers();

}
#endif
#endif

