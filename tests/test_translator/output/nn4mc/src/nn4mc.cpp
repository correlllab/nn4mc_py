
/********************
    nn4mc.cpp

    Code generated using nn4mc.

    This file implements the nerual network and associated functions.

*/
#include "nn4mc.h"
#include <stdlib.h>
#include conv1d.h
#include maxpooling1d.h
#include dense.h


struct Conv1D conv1d;
struct MaxPooling1D max_pooling1d;
struct Conv1D conv1d_1;
struct MaxPooling1D max_pooling1d_1;
struct Dense dense;


void buildLayers(){

    
        conv1d = build_layer_conv1d(&conv1d_W[0], conv1d_b, 3, 1, 50, 1, 32, 0xB, 0x00, 0x00, 1);

       max_pooling1d = build_layer_maxpooling1d(1, 1, 48, 32);

        conv1d_1 = build_layer_conv1d(&conv1d_1_W[0], conv1d_1_b, 3, 1, 48, 32, 32, 0x06, 0x00, 0x00, 1);

       max_pooling1d_1 = build_layer_maxpooling1d(2, 2, 46, 32);

        dense = build_layer_dense(&dense_W[0], dense_b, [46, 32], 1, 0xB);


}


float * fwdNN(float* data)
{

    
        data = fwd_conv1d(conv1d, data);
data = 0xB(data);

        data = fwd_maxpooling1d(max_pooling1d, data);
data = 0xB(data);

        data = fwd_conv1d(conv1d_1, data);
data = 0x06(data);

        data = fwd_maxpooling1d(max_pooling1d_1, data);
data = 0x06(data);

        data = fwd_dense(dense, data);
data = 0xB(data);


    return data;
}

