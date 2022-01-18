
/********************
    nn4mc.cpp

    Code generated using nn4mc.

    This file implements the nerual network and associated functions.

*/
#include "nn4mc.h"
#include <stdlib.h>
#include gru.h
#include dense.h


struct GRU gru;
struct Dense dense;


void buildLayers(){

    
gru = build_layer_gru(
                          &gru_W[0],
                          &gru_Wrec[0],
                          &gru_b[0],
                          0x08,
                          0x07,
                          1,
                          68,
                          10,
);

        dense = build_layer_dense(&dense_W[0], dense_b, 10, 3, 0x06);


}


float * fwdNN(float* data)
{

    
data =  fwd_gru(gru, data);
data = 0x07(data);

        data = fwd_dense(dense, data);
data = 0x06(data);


    return data;
}

