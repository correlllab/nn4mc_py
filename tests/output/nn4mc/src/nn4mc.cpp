
/********************
    nn4mc.cpp

    Code generated using nn4mc.

    This file implements the nerual network and associated functions.

*/

#ifdef __cplusplus
#extern "C" {

#include "neural_network.h"
#include <stdlib.h>

void buildLayers(){

    conv1d_1 = buildConv1D(&conv1d_1_w[0], conv1d_1_b, 4, 1, 50, 2, 8, linear);
conv1d_2 = buildConv1D(&conv1d_2_w[0], conv1d_2_b, 4, 1, 47, 8, 8, linear);
dense_1 = buildDense(&dense_1_w[0], dense_1_b, [44, 8], 64, relu);
dense_2 = buildDense(&dense_2_w[0], dense_2_b, 64, 42, relu);
dense_3 = buildDense(&dense_3_w[0], dense_3_b, 42, 28, relu);
dense_4 = buildDense(&dense_4_w[0], dense_4_b, 28, 18, relu);
dense_5 = buildDense(&dense_5_w[0], dense_5_b, 18, 12, relu);
dense_6 = buildDense(&dense_6_w[0], dense_6_b, 12, 8, relu);
dense_7 = buildDense(&dense_7_w[0], dense_7_b, 8, 5, relu);
dense_8 = buildDense(&dense_8_w[0], dense_8_b, 5, 3, linear);


}


float * fwdNN(float* data)
{

    data = fwdConv1D(conv1d_1, data);
data = fwdConv1D(conv1d_2, data);
data = fwdDense(dense_1, data);
data = fwdDense(dense_2, data);
data = fwdDense(dense_3, data);
data = fwdDense(dense_4, data);
data = fwdDense(dense_5, data);
data = fwdDense(dense_6, data);
data = fwdDense(dense_7, data);
data = fwdDense(dense_8, data);


    return data;
}

}
#endif
