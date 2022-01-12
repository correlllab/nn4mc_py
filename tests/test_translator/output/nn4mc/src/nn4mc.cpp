
/********************
    nn4mc.cpp

    Code generated using nn4mc.

    This file implements the nerual network and associated functions.

*/

#ifdef __cplusplus
extern "C" {

#include "nn4mc.h"
#include <stdlib.h>

void buildLayers(){

    
gru = build_layer_gru(
                          &dense_W[0],
                          &gru_Wrec[0],
                          &dense_b[0],
                          sigmoid,
                          tanh,
                          1,
                          68,
                          1,
);

        dense = build_layer_dense(&dense_W[0], dense_b, 1, 3, relu);


}


float * fwdNN(float* data)
{

    
data =  fwd_gru(gru, data);
data = tanh(data);

        data = fwd_dense(dense, data);
data = relu(data);


    return data;
}

}
#endif
