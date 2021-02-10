/********************
    maxpool1d.h

    Code generated using nn4mc.

    This file defines an 1 dimensional maximum pooling layer.

*/

#ifndef __MAXPOOL1D_H__
#define __MAXPOOL1D_H__
#include <math.h>
#include <stdlib.h>

struct MaxPooling1D {
	int pool_size;
	int strides;
    char padding;
	int input_shape[2];
	int output_shape[2];
};

float * padding_1d(struct MaxPooling1D, float *);

struct MaxPooling1D build_layer_maxpooling1d(int, int, int, int, char);

float * fwd_maxpooling1d(struct MaxPooling1D, float * );

#endif
