
#include "neural_network.h"
#include <stdlib.h>

void buildLayers(){

    conv1d_1 = buildConv1D(&[0], , 4, 1, 4, 2, 8, linear);
conv1d_2 = buildConv1D(&[0], , 4, 1, 4, 8, 8, linear);
dense_1 = buildDense(&[0], , 352, 0, relu);
dense_2 = buildDense(&[0], , 64, 0, relu);
dense_3 = buildDense(&[0], , 42, 0, relu);
dense_4 = buildDense(&[0], , 28, 0, relu);
dense_5 = buildDense(&[0], , 18, 0, relu);
dense_6 = buildDense(&[0], , 12, 0, relu);
dense_7 = buildDense(&[0], , 8, 0, relu);
dense_8 = buildDense(&[0], , 5, 0, linear);


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

