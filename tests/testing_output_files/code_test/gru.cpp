/********************
    gru.cpp

    Code generated using nn4mc.

    This file implements a gru layer.

*/
#include "gru.h"

struct GRU build_layer_gru(const float* W, const float* U, const float* b,
                          char recurrent_activation, char activation,
                          int input_shape_0, int input_shape_1, int output_units
){
	struct GRU layer;

	layer.weights = W;
    layer.big_u = U;
	layer.biases = b;

	layer.weight_shape[0] = input_shape_1;
	layer.weight_shape[1] = output_units;

    layer.big_u_shape[0] = output_units;
    layer.big_u_shape[1] = output_units;

    layer.biases_shape[0] = 2;
    layer.biases_shape[1] = output_units;

    layer.h_tm1 = (float*)malloc(output_units * sizeof(float));
    for (int i = 0 ; i < output_units; i++) {
        layer.h_tm1[i] = 0.0;
    }

    layer.recurrent_activation = recurrent_activation;
    layer.activation = activation;

    layer.input_shape[0] = input_shape_0;
    layer.input_shape[1] = input_shape_1;

    layer.output_shape[0] = output_units;

	return layer;
}

float * fwd_gru(struct GRU L, float * input)
{
    float* z_t = (float*)malloc(L.output_shape[0] * sizeof(float));
    float* r_t = (float*)malloc(L.output_shape[0] * sizeof(float));
    float* h_hat_t = (float*)malloc(L.output_shape[0]*sizeof(float));
    float* h_t = (float*)malloc(L.output_shape[0]*sizeof(float));

    for (int i =0; i < L.output_shape[0]; i++){
        z_t[i] = L.biases[i];
        r_t[i] = L.biases[i + L.output_shape[0]];
        h_hat_t[i] = L.biases[i + 2*L.output_shape[0]];
        h_t[i] = 0.;
    }

    for (int i = 0; i < L.output_shape[0]; i++){
        for (int j = 0; j < L.input_shape[1]; j++){
            z_t[i] += L.weights[i*L.output_shape[0]+ j] * input[j];
            r_t[i] += L.weights[i*L.output_shape[0] + (j + L.output_shape[0])] * input[j];
            h_hat_t[i] += L.weights[i*L.output_shape[0] + (j + 2*L.output_shape[0])] * input[j];
        }
    }

    for (int i = 0; i < L.output_shape[0]; i++){
        for (int j = 0; j < L.output_shape[0]; j++){
            z_t[i] += L.big_u[i*L.output_shape[0] + j] * L.h_tm1[j];
            r_t[i] += L.big_u[i*L.output_shape[0] + (j + L.output_shape[0])] * L.h_tm1[j];
        }
    }

    z_t = activate(z_t, L.output_shape[0], L.activation);
    r_t = activate(r_t, L.output_shape[0], L.activation);

    for (int i = 0; i < L.output_shape[0]; i++){
        for (int j = 0; j < L.output_shape[0]; j++){
            h_hat_t[i] += L.big_u[i*L.output_shape[0] + (j + 2*L.output_shape[0])] * r_t[j] * L.h_tm1[j];
        }
    }

    h_hat_t = activate(h_hat_t, L.output_shape[0], L.recurrent_activation);

    for (int i = 0; i < L.output_shape[0]; i++){
        h_t[i] = (1 - z_t[i]) * L.h_tm1[i] + z_t[i] * h_hat_t[i];
        L.h_tm1[i] = h_t[i];
    }

    free(z_t);
    free(r_t);
    free(h_hat_t);
    free(input);
    return h_t;
}