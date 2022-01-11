
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
                          &<%RECURRENT_WEIGHT_NAME>[0],
                          &<%BIAS_NAME>[0],
                          <%RECURRENT_ACTIVATION>,
                          <%ACTIVATION>,
                          <%INPUT_SHAPE_0>,
                          <%INPUT_SHAPE_1>,
                          <%OUTPUT_SHAPE>,
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
