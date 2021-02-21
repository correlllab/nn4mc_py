/********************
    maxpool2d.cpp

    Code generated using nn4mc.

    This file implements a 1 dimensional maximum pooling layer.

*/
#include "maxpooling2d.h"

#define max(a, b) (((a)>(b) ? (a) : (b)))
#define min(a, b) (((a)<(b) ? (a) : (b)))

struct MaxPooling2D build_layer_maxpooling2d(int pool_size_0, int pool_size_1, int strides_0, int strides_1, char padding, int input_shape_0, int input_shape_1, int input_shape_2)
{
	MaxPooling2D layer;

	layer.strides[0] = strides_0;
    layer.strides[1] = strides_1;

    layer.pool_size[0] = pool_size_0;
    layer.pool_size[1] = pool_size_1;
    layer.input_shape[0] = input_shape_0;
    layer.input_shape[1] = input_shape_1;
    layer.input_shape[2] = input_shape_2;

	layer.output_shape[0] = (int)floor((input_shape_0 - pool_size_0) / strides_0) + 1;
	layer.output_shape[1] = (int)floor((input_shape_1 - pool_size_1) / strides_1) + 1;
    layer.output_shape[2] = input_shape_2;

    if (padding == 0x03)
    {
        layer.output_shape[0] = (int)ceil(input_shape_0 / strides_0);
        layer.output_shape[1] = (int)ceil(input_shape_1 / strides_1);
        layer.output_shape[2] = input_shape_2;
    }

	return layer;
}


float * fwd_maxpooling2d(struct MaxPooling2D L, float * input)
{

    float * h = (float*)malloc((int)L.output_shape[0]*L.output_shape[1]*L.output_shape[2] * sizeof(float));

	for (int i = 0; i < L.output_shape[0]; i++)
	{
		for (int j = 0; j < L.output_shape[1]; j++)
		{
            for (int k = 0; k < L.output_shape[2]; k++)
            {
                int idx = (i*L.output_shape[1] + j) * L.output_shape[2] + k;

                h[idx] = -INFINITY;

                for (int s1 = 0; s1 < L.pool_size[0]; s1++)
                {
                    for (int s2 = 0; s2 < L.pool_size[1]; s2++)
                    {
                        float x = *(input + ((L.strides[0] * i + s1) * L.input_shape[1] + (L.strides[1] * j + s2)) * L.input_shape[2] + k);
                        h[idx] = max(x, h[idx]);
                    }
                }
            }
        }
    }
    //free(input);
    return h;
}

