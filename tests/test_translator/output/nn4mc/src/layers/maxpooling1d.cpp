
/********************
    maxpool1d.cpp

    Code generated using nn4mc.

    This file implements a 2 dimensional maximum pooling layer.

*/

#include "maxpooling1d.h"

#define max(a, b) (((a)>(b) ? (a) : (b)))
#define min(a, b) (((a)<(b) ? (a) : (b)))

struct MaxPooling1D build_layer_maxpooling1d(int pool_size, int strides, int input_sh0, int input_sh1, char padding)
{
	struct MaxPooling1D layer;

	layer.strides = strides;
    layer.pool_size = pool_size;

    layer.input_shape[0] = input_sh0;
    layer.input_shape[1] = input_sh1;

    layer.padding = padding;

	layer.output_shape[0] = (int)(ceil((input_sh0 - pool_size + 1) / strides)) + 1;
	layer.output_shape[1] = (int)input_sh1;

	return layer;
}

float * padding_1d(struct MaxPooling1D L, float * input){

        if (L.padding == 0x03) { // padding is same
              int input_size = (int)(L.input_shape[0] * L.input_shape[1]);
              int pad = (int)(L.pool_size / 2);
              input_size += (int)(L.input_shape[0] * pad);
              float new_input[input_size];
              int left_pad = floor(pad / 2);
              int right_pad = abs(pad - left_pad);
              for (int i = 0; i < input_size; i++) new_input[i] = 0.0;
              for (int i = 0; i < L.input_shape[0]; i++){
                  for (int j = 0; j < L.input_shape[1]; j++){
                      new_input[(L.input_shape[1]) * i + j] = input[L.input_shape[1] * i + j];
                  }
              }

              input = (float*)realloc(input, input_size * sizeof(float));
              for (int i = 0; i < input_size; i++) input[i] = new_input[i];
        }

    return input;
}

float * fwd_maxpooling1d(struct MaxPooling1D L, float * input)
{
    input = padding_1d(L, input);

    float * h = (float*)malloc(L.output_shape[0]*L.output_shape[1] * sizeof(float));

    for (int j = 0; j < L.output_shape[1]; j++)
    {
        for (int i = 0; i < L.output_shape[0]; i++)
        {
            int idx = i * L.output_shape[1] + j;

            h[idx] = -INFINITY;
            for (int s = 0; s < L.pool_size; s++){
                float x = input[(L.strides*i + s)*L.input_shape[1] + j];
                h[idx] = max(x, h[idx]);
            }
		}
	}

    free(input);
    return h;
}

