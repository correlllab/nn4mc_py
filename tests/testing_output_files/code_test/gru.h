/********************
    gru.h

    Code generated using nn4mc.

    This file defines a gru model as implemented by Keras.

*/
#ifndef __GRU_H__
#define __GRU_H__
#include "activations.h"
#include <math.h>
#include <stdlib.h>

struct GRU {

	const float* weights;			    // Pointer to constant weight array
	const float* big_u;
	const float* biases;				// Pointer to constant bias

    int weight_shape[2];
    int big_u_shape[2];
    int biases_shape[2];

    char recurrent_activation;
    char activation;

	int input_shape[2];		            // (TIMESTEPS, FEATURE)
	int output_shape[1];	            // (UNITS)

    float* h_tm1;                       // storing past h value
};

struct GRU build_layer_gru(const float*, const float*, const float*,
                            char, char, int, int, int);

float * fwd_gru(struct GRU, float*);

#endif
