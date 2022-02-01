
/********************

    gru.cpp

    Code generated using nn4mc.

    This file implements a gru layer.

    Implementation from:
    Cho et. al (2014)
    https://arxiv.org/pdf/1406.1078.pdf
    https://github.com/keras-team/keras/blob/0a726d9844b4dce3efc5077718bdfb905e1750fa/keras/layers/recurrent.py#L1861

*********************/
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
    const int M = L.output_shape[0];
    const int NM = L.input_shape[1] * L.output_shape[0];
    float * x_z = (float*)malloc(NM * sizeof(float));
    float * x_r = (float*)malloc(NM * sizeof(float));
    float * x_h = (float*)malloc(NM * sizeof(float));
    float * h_t = (float*)malloc(NM * sizeof(float));
    for (int i = 0; i < NM; i++){
        x_z[i] = 0.0;
        x_r[i] = 0.0;
        x_h[i] = 0.0;
        h_t[i] = 0.0;
    }
    for (int i = 0; i < M; i++){
        for (int k = 0; k < L.input_shape[0]; k++){
            x_z[k * M + i] += L.biases[i];
            x_r[k * M + i] += L.biases[i + M];
            x_h[k * M + i] += L.biases[i + 2 * M];
            x_z[k * M + i] += L.biases[i + 3 * M];
            x_r[k * M + i] += L.biases[i + 4 * M];
            x_h[k * M + i] += L.biases[i + 5 * M];
            for (int j = 0; j < L.input_shape[0]; j++){
                x_z[k * M + i] += input[k * L.input_shape[1] + j] * *(L.weights + j * 3*M + i);
                x_r[k * M + i] += input[k * L.input_shape[1] + j] * *(L.weights + j * 3*M + i + M);
                x_h[k * M + i] += input[k * L.input_shape[1] + j] * *(L.weights + j * 3*M + i + 2*M);
            }
            for (int j = 0; j < M; j++){
                x_z[k * M + j] += *(L.big_u + i * 3 * M + j) * h_tm1[j];
                x_r[k * M + j] += *(L.big_u + i * 3 * M + j + M) * h_tm1[j];
            }
        }

    }
    free(input);
    x_z = activate(x_z, NM, L.recurrent_activation);
    x_r = activate(x_r, NM, L.recurrent_activation);
    for (int i = 0; i < L.input_shape[0]; i++){
        for (int j = 0; j < M; j++){
            for (int k = 0; k < M; k++){
                x_h[i * M + j] += *(L.big_u + k * 3 * M + j + 2 * M) *
                                                        h_tm1[j] * x_r[i * M + k];
            }
        }
    }
    x_h = activate(x_h, NM, L.activation);
    for (int i = 0; i < M; i++){
        for (int j = 0; j < L.input_shape[0]; j++){
                h_t[j * M + i] = (1.0 - x_z[j * M + i]) *
                                        x_h[j * M + i] +
                                            x_z[j * M + i] *
                                            h_tm1[i];

        }
    }
    for (int i = 0; i < L.output_shape[0]; i++){
        h_tm1[i] = h_t[i];
    }
    free(x_h);
    free(x_r);
    free(x_z);
    return h_t;
}
