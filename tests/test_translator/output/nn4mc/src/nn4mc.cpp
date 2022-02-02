
/********************
    nn4mc.cpp

    Code generated using nn4mc.

    This file implements the nerual network and associated functions.

*/
#include "nn4mc.h"
#include <stdlib.h>
#include "conv1d.h"
#include "gru.h"
#include "dense.h"


struct Conv1D conv1d_1;
struct GRU gru_1;
struct Dense dense_3;
struct Dense dense_4;
struct Dense dense_5;


void buildLayers(){

    
        conv1d_1 = build_layer_conv1d(&conv1d_1_W[0], conv1d_1_b, 2, 1, 2, 100, 32, 0xB, 0x00, 0x00, 1);

gru_1 = build_layer_gru(
                          &gru_1_W[0],
                          &gru_1_Wrec[0],
                          &gru_1_b[0],
                          0x08,
                          0x07,
                          1,
                          32,
                          20
);

        dense_3 = build_layer_dense(&dense_3_W[0], dense_3_b, 20.0, 10, 0x06);

        dense_4 = build_layer_dense(&dense_4_W[0], dense_4_b, [10], 1, 0x07);

        dense_5 = build_layer_dense(&dense_5_W[0], dense_5_b, [1], 6, 0x00);


}


float * fwdNN(float* data)
{

    
        data = fwd_conv1d(conv1d_1, data);

data =  fwd_gru(gru_1, data);

        data = fwd_dense(dense_3, data);

        data = fwd_dense(dense_4, data);

        data = fwd_dense(dense_5, data);


    return data;
}

