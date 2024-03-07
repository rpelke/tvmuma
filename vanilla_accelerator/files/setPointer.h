#ifndef SET_POINTER_H
#define SET_POINTER_H

#include <tvmgen_default.h>

void setPointer(struct tvmgen_default_inputs* input_struct, void* shmp_start, const int TVM_DEFAULT_INPUT_SIZE, const int TVM_DEFAULT_WEIGHTS_SIZE[], const int TVM_DEFAULT_DENSE_SIZE[]) {
    input_struct->input = shmp_start;
    input_struct->weight0 = input_struct->input  + TVM_DEFAULT_INPUT_SIZE;
    input_struct->weight1 = input_struct->weight0 + TVM_DEFAULT_WEIGHTS_SIZE[0];
    input_struct->weight2 = input_struct->weight1 + TVM_DEFAULT_WEIGHTS_SIZE[1];
    input_struct->weight3 = input_struct->weight2 + TVM_DEFAULT_WEIGHTS_SIZE[2];
    input_struct->dense0 = input_struct->weight3 + TVM_DEFAULT_WEIGHTS_SIZE[3];
    input_struct->dense1 = input_struct->dense0 + TVM_DEFAULT_DENSE_SIZE[0];
}

#endif // SET_POINTER_H
