
/********************
    conv2d.cpp

    Code generated using nn4mc.

    This file implements a 2 dimensional convolution layer.

*/

#include "conv2d.h"
#include "activations.h"
#include <stdlib.h>
#include <math.h>

#define max(a, b) (((a)>(b) ? (a) : (b)))
#define min(a, b) (((a)<(b) ? (a) : (b)))

struct Conv2D build_layer_conv2d(const float* W, const float* b, int kernel_shape_0, int kernel_shape_1, int filters, int strides_0, int strides_1, int input_shape_0,int input_shape_1,int input_shape_2, char activation, char padding, char data_format, int dilation_rate_0, int dilation_rate_1)
{
	struct Conv2D layer;

	layer.weights = W;
	layer.bias = b;

	layer.weight_shape[0] = kernel_shape_0;
	layer.weight_shape[1] = kernel_shape_1;
	layer.weight_shape[2] = input_shape_2;
	layer.weight_shape[3] = filters;

    layer.filters = filters;
    layer.activation = activation;
    layer.padding = padding;
    layer.data_format = data_format;

    layer.kernel_shape[0] = kernel_shape_0;
    layer.kernel_shape[1] = kernel_shape_1;
   
    layer.dilation_rate[0] = dilation_rate_0;
    layer.dilation_rate[1] = dilation_rate_1;

	layer.strides[0] = strides_0;
	layer.strides[1] = strides_1;

	layer.input_shape[0] = input_shape_0;
	layer.input_shape[1] = input_shape_1;
	layer.input_shape[2] = input_shape_2;

	layer.output_shape[0] = (int)((input_shape_0 - kernel_shape_0) / strides_0) + 1;
	layer.output_shape[1] = (int)((input_shape_1 - kernel_shape_1) / strides_1) + 1;
	layer.output_shape[2] = (int)(filters);

	return layer;
}

float * padding_2d(struct Conv2D L, float * input, int * shape_0_change, int * shape_1_change){
        if (L.padding == 0x03) { // padding is same
            int input_size = (int)(L.input_shape[0] * L.input_shape[1] * L.input_shape[2]);

            int pad = (int)(L.filters / 2);
            int pad_0 = floor(pad / 2);
            int pad_1 = abs(pad - pad_0);

            *shape_0_change = (int)floor((pad_0 + pad_1) / L.strides[0]);
            *shape_1_change = (int)floor((pad_0 + pad_1) / L.strides[1]);

            input_size += (int)(2*pad_0 + 2*pad_1 + 2*pad_0*pad_1 + pad_0*pad_0 + pad_1*pad_1)*L.input_shape[2];
            float new_input[input_size];
            for(int i = 0; i < input_size; i++) new_input[i] = 0.0;

            for (int i = 0; i < L.input_shape[0]; i++){
                for (int j = 0; j < L.input_shape[1]; j++){
                    for (int k = 0; k < L.input_shape[2]; k++){
                        new_input[((i + pad_0) * (L.input_shape[1] + *shape_1_change) + (j + pad_0)) * L.input_shape[2] + k] = *(input + (i * L.input_shape[1] + j) * L.input_shape[2] + k);
                    }
                }
            }
            //free(input);
            input = (float*)malloc(input_size*sizeof(float));
            for (int i = 0; i < input_size; i++) input[i] = new_input[i];

        }
    return input;
}

float* fwd_conv2d(struct Conv2D L, float* input)
{
    int shape_0_ch = 0 , shape_1_ch = 0;

    input = padding_2d(L, input, &shape_0_ch, &shape_1_ch);

    int output_size = (L.output_shape[0] + shape_0_ch) * (L.output_shape[1] + shape_1_ch) * L.output_shape[2];

    float * h = (float*)malloc(output_size * sizeof(float));

    for (int i = 0; i < (int)((L.output_shape[0] + shape_0_ch)); i++)
    {
        for (int j = 0; j < (int)((L.output_shape[1] + shape_1_ch)); j++)
        {
           for (int k = 0; k < L.output_shape[2]; k++)
           {
				int idx = (i * (L.output_shape[1] + shape_1_ch) + j) * L.output_shape[2] + k;
				h[idx] = L.bias[k];

				for (int k_x = 0; k_x < L.weight_shape[0]; k_x++)
					for (int k_y = 0; k_y < L.weight_shape[1]; k_y++)
					    for (int k_z = 0; k_z < L.weight_shape[2]; k_z++)
                            h[idx] += *(L.weights + ((k_x * L.weight_shape[1] + k_y) * L.weight_shape[2] + k_z) * L.weight_shape[3] + k) * *(input + ((k_x + i * L.strides[0]) * (L.input_shape[1] + shape_1_ch) + (k_y + j*L.strides[1])) * L.input_shape[2] + k_z);
            }
        }
    }
	h = activate(h, output_size, L.activation);
    //free(input);
    return h;
}

