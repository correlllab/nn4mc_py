
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


struct Conv1D conv1d;
struct GRU gru;
struct Dense dense;
struct Dense dense_1;
struct Dense dense_2;


void buildLayers(){

    
        conv1d = build_layer_conv1d(&conv1d_W[0], conv1d_b, 2, 1, 2, 30, 12, 0xB, 0x00, 0x00, 1);

gru = build_layer_gru(
                          &gru_W[0],
                          &gru_Wrec[0],
                          &gru_b[0],
                          0x08,
                          0x07,
                          1,
                          12,
                          50
);

        dense = build_layer_dense(&dense_W[0], dense_b, 50.0, 10, 0x06);

        dense_1 = build_layer_dense(&dense_1_W[0], dense_1_b, [10], 1, 0x07);

        dense_2 = build_layer_dense(&dense_2_W[0], dense_2_b, [1], 6, 0x00);


}


float * fwdNN(float* data)
{

    
        data = fwd_conv1d(conv1d, data);

data =  fwd_gru(gru, data);

        data = fwd_dense(dense, data);

        data = fwd_dense(dense_1, data);

        data = fwd_dense(dense_2, data);


    return data;
}

