
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

	layer.output_shape[0] = (int)(input_shape_0 - kernel_shape_0 + 1);
	layer.output_shape[1] = (int)(input_shape_1 - kernel_shape_1 + 1);
	layer.output_shape[2] = (int)(filters);

	return layer;
}

float * padding_2d(struct Conv2D L, float * input){
        if (L.padding == 0x03) { // padding is same
            int input_size = (int)(L.input_shape[0] * L.input_shape[1] * L.input_shape[2]);
            int pad = (int)(L.filters / 2);
            int pad_0 = floor(pad / 2);
            int pad_1 = abs(pad - pad_0);
            input_size += (int)(2*pad_0 + 2*pad_1 + 2*pad_0*pad_1 + pad_0*pad_0 + pad_1*pad_1);
            float new_input[input_size];

            for (int i = 0; i < L.input_shape[0]; i++){
                for (int j = 0; j < L.input_shape[1]; j++){
                    for (int k = 0; k < L.input_shape[2]; k++){
                        new_input[(i * (L.input_shape[1] + pad_0) + pad_0*(L.input_shape[1] + pad_1) + j) * L.input_shape[2] + k] = input[(i * L.input_shape[1] + j) * L.input_shape[2] + k];
                    }
                }
            }
            input = (float*)realloc(input, input_size * sizeof(float));
            for (int i = 0; i < input_size; i++) input[i] = new_input[i];
        }
    return input;
}

float* fwd_conv2d(struct Conv2D L, float* input)
{
    input = padding_2d(L, input);

    int output_size = L.output_shape[0]*L.output_shape[1]*L.output_shape[2];

    float * h = (float*)malloc(output_size * sizeof(float));

	for(int i = 0; i < L.output_shape[0]; i++)
	{
		for(int j = 0; j < L.output_shape[1]; j++)
		{
			for(int k = 0; k < L.output_shape[2]; k++)
			{
				int output_idx = i * L.output_shape[1] * L.output_shape[2] + j * L.output_shape[2] + k;

				h[output_idx] = L.bias[k];

				for(int kernel_position_x = 0; kernel_position_x < L.kernel_shape[0]; kernel_position_x++)
				{

                    int mm = L.kernel_shape[0] - 1 - kernel_position_x;

					for(int kernel_position_y = 0; kernel_position_y < L.kernel_shape[1]; kernel_position_y++)
					{

                        for (int f=0; f<L.filters; f++){
                        int nn = L.kernel_shape[1] - 1 - kernel_position_y;

                        int ii = i + (L.kernel_shape[1]/2 - mm);
                        int jj = j + (L.kernel_shape[0]/2 - nn);

                        if (ii >= 0 && ii < i && jj >= 0 && jj < j){
                            h[output_idx] += *(L.weights + L.weight_shape[3]*L.weight_shape[2]*L.weight_shape[1]*kernel_position_x + L.weight_shape[2]*L.weight_shape[1]* kernel_position_y + L.weight_shape[1]*f) * (input[L.input_shape[1]*L.input_shape[2]*ii + L.input_shape[2] * jj]);
                       }
                   }
                }
            }

			}
		}
	}

	h = activate(h, output_size, L.activation);
    //free(input);
    return h;
}

