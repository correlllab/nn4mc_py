/********************
    conv1d.cpp

    Code generated using nn4mc.

    This file implements a 1 dimensional convolution layer.

*/

#include "conv1d.h"
#include "activations.h"
#include <math.h>
#include <stdlib.h>

#define max(a, b) (((a)>(b) ? (a) : (b)))
#define min(a, b) (((a)<(b) ? (a) : (b)))

struct Conv1D build_layer_conv1d(const float* W, const float* b, int kernel_size, int strides, int input_sh0, int input_sh1, int filters, char activation, char padding, char data_format, int dilation_rate)
{
	struct Conv1D layer;

	layer.weights = W;
	layer.biases = b;

	layer.weight_shape[0] = kernel_size;
	layer.weight_shape[1] = input_sh1;
	layer.weight_shape[2] = filters;

	layer.strides = strides;
    layer.kernel_shape[0] = kernel_size;
	
    layer.input_shape[0] = input_sh0;
	layer.input_shape[1] = input_sh1;
    
    layer.dilation_rate = dilation_rate;

    layer.activation = activation;
    layer.padding = padding;
    layer.data_format = data_format;
    
    layer.filters = filters;

	layer.output_shape[0] = (int)((layer.input_shape[0] - layer.kernel_shape[0]) / layer.strides + 1);
	layer.output_shape[1] = (int)layer.filters;

	return layer;
}

float * padding_1d(struct Conv1D L, float * input){
        if (L.padding == 0x02){ // padding is causal (tested)
               int input_size = (int)(L.input_shape[0] * L.input_shape[1]);
               int left_pad = (int)(L.dilation_rate * (L.kernel_shape[0] - 1));
               input_size += (int)(left_pad * L.input_shape[0]);
               float new_input[input_size];
               for (int i = 0; i < input_size; i++) new_input[i] = 0.0;
               for (int i = 0; i < L.input_shape[0]; i++){
                   for (int j = 0; j < L.input_shape[1]; j++){
                       new_input[i * (L.input_shape[1] + left_pad) + j + left_pad] = input[j + L.input_shape[1] * i];
                   }
               }
               input = (float*)realloc(input, input_size * sizeof(float));
               for (int i = 0; i < input_size; i++) input[i] = new_input[i];
        }

        if (L.padding == 0x03) { // padding is same
              int input_size = (int)(L.input_shape[0] * L.input_shape[1]);
              int pad = (int)(L.filters / 2);
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


float * fwd_conv1d(struct Conv1D L, float * input)
{
    input = padding_1d(L, input);

    int input_size = (int)(sizeof(input) / sizeof(input[0]));

    if (L.data_format == 0x02){
        for (int i = 0; i < L.input_shape[0]; i++){
            for (int j =0 ; j<L.input_shape[1]; j++){
                float temp = input[i + L.input_shape[1] * j];
                input[i + L.input_shape[1] * j] = input[j+L.input_shape[1]*i];
                input[j + L.input_shape[1] * i] = temp;
            }
        } 
    }

    float old_input[input_size];
    for (int i = 0; i < input_size; i++) old_input[i] = input[i];
    //int output_size =(int)(L.output_shape[0]*L.output_shape[1]);
    int output_size = (int)(input_size);

    input = (float*)realloc(input, output_size * sizeof(float));
    for (int i=0; i < output_size; i++) input[i] = 0.0;

	for(int i = 0; i < L.output_shape[0]; i++)
	{
		for(int j = 0; j < L.output_shape[1]; j++)
		{
            int idx = i*L.output_shape[1] + j;

			input[idx] = L.biases[j];

			for(int x = 0; x < L.kernel_shape[0]; x++)
			{
				for(int y = 0; y < L.weight_shape[1]; y++)
				{
                    input[idx] += *(L.weights + x*L.weight_shape[1]*L.weight_shape[2] + y*L.weight_shape[2] +  j) * old_input[(i+x)*L.input_shape[1] +  y];
				}
			}

		    input = activate(input,input_size+1,L.activation);
		}
	}

    return input;
}

