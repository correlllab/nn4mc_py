/********************
    maxpool2d.h

    Code generated using nn4mc.

    This file defines an 2 dimensional maximum pooling layer.

*/
#ifndef __MAXPOOLING2D_H__
#define __MAXPOOLING2D_H__

#include <math.h>
#include <stdlib.h>

struct MaxPooling2D {
	int pool_size[2];
	int strides[2];
	int input_shape[3];
	int output_shape[3];
};

struct MaxPooling2D build_layer_maxpooling2d(int, int, int, int, char, int, int, int);

float * fwd_maxpooling2d(struct MaxPooling2D , float *);

#endif
