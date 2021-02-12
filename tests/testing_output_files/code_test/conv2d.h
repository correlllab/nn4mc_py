
/********************
    conv2d.h

    Code generated using nn4mc.

    This file defines an 2 dimensional convolution layer.

*/

#ifndef __CONV2D_H__
#define __CONV2D_H__

struct Conv2D {
	const float* weights;
	const float* bias;
	int weight_shape[4];
    int strides[2];
    int filters;
    int dilation_rate[2];
    char activation;
    char padding;
    char data_format;
    int kernel_shape[2];
	int input_shape[3];
	int output_shape[3];
};

struct Conv2D build_layer_conv2d(const float*, const float*, int, int, int, int, int, int, int, int, char,char,char, int , int);

float* fwd_conv2d(struct Conv2D, float*);

float * padding_2d(struct Conv2D, float *, int*, int*);

#endif
