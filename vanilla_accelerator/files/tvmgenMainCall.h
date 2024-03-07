#ifndef MAIN_CALL_H
#define MAIN_CALL_H

#include <tvmgen_default.h>
#include <stdint.h>

void callTVMmain(struct tvmgen_default_inputs *tvmgen_default_inputs, struct tvmgen_default_outputs *tvmgen_default_outputs, uint8_t* workspace) {
    tvmgen_default___tvm_main__(
        tvmgen_default_inputs->input,
        tvmgen_default_inputs->weight0,
        tvmgen_default_inputs->weight1,
        tvmgen_default_inputs->weight2,
        tvmgen_default_inputs->weight3,
        tvmgen_default_inputs->dense0,
        tvmgen_default_inputs->dense1,
        tvmgen_default_outputs->output,
        workspace);
}

#endif // MAIN_CALL_H
