
/********************
    Dense.h

    Code generated using nn4mc.

    This file defines a densely connected layer.

*/

#ifndef __DENSE_H__
#define __DENSE_H__


struct Dense {

	const float* weights;
	const float* biases;

    char activation;

    int weight_shape[2];

	int input_shape[1];
	int output_shape[1];
};

struct Dense build_layer_dense(const float*, const float*, int, int, char);

float * fwd_dense(struct Dense, float* );

#endif
