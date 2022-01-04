/********************
    gru.cpp

    Code generated using nn4mc.

    This file implements a gru layer.

    Implementation from:
    Cho et. al (2014)
    https://arxiv.org/pdf/1406.1078.pdf
    https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py

*/
#include "gru.h"

struct GRU build_layer_gru(
                          const float* W,
                          const float* U,
                          const float* b,
                          char recurrent_activation,
                          char activation,
                          int input_shape_0,
                          int input_shape_1,
                          int output_units
){
	struct GRU layer;

	layer.weights = W;
    layer.big_u = U;
	layer.biases = b;

	layer.weight_shape[0] = input_shape_1;
	layer.weight_shape[1] = 3 * output_units;

    layer.big_u_shape[0] = output_units;
    layer.big_u_shape[1] = 3 * output_units;

    layer.biases_shape[0] = 2;
    layer.biases_shape[1] = 3 * output_units;

    layer.recurrent_activation = recurrent_activation;
    layer.activation = activation;

    layer.input_shape[0] = input_shape_0;
    layer.input_shape[1] = input_shape_1;

    layer.output_shape[0] = output_units;

    layer.h_tm1 = (float*)malloc(output_units * sizeof(float));
    for (int i = 0; i < output_units; i++){
        layer.h_tm1[i] = 0.0;
    }
	return layer;
}

float * fwd_gru(struct GRU L, float * input)
{
    float* x_z = (float*)malloc(L.output_shape[0] * sizeof(float));
    float* x_r = (float*)malloc(L.output_shape[0] * sizeof(float));
    float* x_h = (float*)malloc(L.output_shape[0] * sizeof(float));
    float* h_t = (float*)malloc(L.output_shape[0] * sizeof(float));

    for (int i = 0; i < L.output_shape[0]; i++){
        x_z[i] = *(L.biases + i);
        x_r[i] = *(L.biases + i + L.output_shape[0]);
        x_h[i] = *(L.biases + i + 2 * L.output_shape[0]);
        x_z[i] += *(L.biases + 3 * L.output_shape[0] + i);
        x_r[i] += *(L.biases + 4 * L.output_shape[0] + i);
        x_h[i] += *(L.biases + 5 * L.output_shape[0] + i);
        for (int j = 0; j < L.input_shape[0]; j++){
            for (int k = 0; k < L.input_shape[1]; k++){
                int idx = k * L.input_shape[1] + j;
                x_z[i] += *(L.weights + i * L.weight_shape[1] + j) * input[idx];
                x_r[i] += *(L.weights + (i + L.output_shape[0]) *
                                L.weight_shape[1] + j) * input[idx];
                x_h[i] += *(L.weights + (i + 2 * L.output_shape[0]) *
                               L.weight_shape[1] + j) * input[idx];
            }
        }
        for (int j = 0; j < L.output_shape[0]; j++){
            x_z[i] += *(L.big_u + i * L.big_u_shape[1] + j) * L.h_tm1[j];
            x_r[i] += *(L.big_u + i * L.big_u_shape[1] + j +
                                L.output_shape[0]) * L.h_tm1[j];
            x_h[i] += *(L.big_u + i * L.big_u_shape[1] + j +
                                2 * L.output_shape[0]) * L.h_tm1[j];
        }
    }
    x_z = activate(x_z, L.output_shape[0], L.recurrent_activation);
    x_r = activate(x_r, L.output_shape[0], L.recurrent_activation);

    for (int i = 0; i < L.output_shape[0]; i++){
        for (int j = 0; j < L.output_shape[0]; j++){
            x_h[i] += *(L.big_u + i * L.big_u_shape[0] +
                                    (j + 2 * L.output_shape[0])) * L.h_tm1[i];
        }
        x_h[i] *= x_r[i];
    }
    x_h = activate(x_h, L.output_shape[0], L.activation);

    for (int i = 0; i < L.output_shape[0]; i++){
        h_t[i] = (1. - x_z[i]) * L.h_tm1[i] + x_z[i] * x_h[i];
        L.h_tm1[i] = h_t[i];
    }
    //free(z_t);
    //free(r_t);
    //free(h_hat_t);
    //free(input);
    return h_t;
}