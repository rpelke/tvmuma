#ifndef SET_STRUCT_H
#define SET_STRUCT_H

#include <tvmgen_default.h>

#include "tvmgen_default_input_data_input.h"

#include "tvmgen_default_input_data_weight0.h"

#include "tvmgen_default_input_data_weight1.h"

#include "tvmgen_default_input_data_weight2.h"

#include "tvmgen_default_input_data_weight3.h"

#include "tvmgen_default_input_data_dense0.h"

#include "tvmgen_default_input_data_dense1.h"

void setStruct(struct tvmgen_default_inputs* input_struct) {
    input_struct->input = tvmgen_default_input_data_input;
    input_struct->weight0 = tvmgen_default_input_data_weight0;
    input_struct->weight1 = tvmgen_default_input_data_weight1;
    input_struct->weight2 = tvmgen_default_input_data_weight2;
    input_struct->weight3 = tvmgen_default_input_data_weight3;
    input_struct->dense0 = tvmgen_default_input_data_weight0;
    input_struct->dense1 = tvmgen_default_input_data_weight1;
}

#endif // SET_STRUCT_H
