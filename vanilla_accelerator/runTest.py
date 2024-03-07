# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from tvm.micro.testing.aot_test_utils import AOT_DEFAULT_RUNNER
import tvm
from tvm import relay, te
from backend import VanillaAcceleratorBackend
from tvm.relay import transform
from collections import OrderedDict
import numpy as np


from tvm.testing.aot import (
    AOTTestModel as AOTModel,
    AOTTestRunner as AOTRunner,
    generate_ref_data,
    compile_and_run,
)


def create_conv2d(groups=1, runner=AOT_DEFAULT_RUNNER, weight_shape=32):
    dtype = "float32"
    ishape = (1, 32, 14, 14)
    wshape = (32, weight_shape, 3, 3)
    pass_config = {"tir.usmp.enable": True}
    runner = AOTRunner(
        makefile=runner.makefile,
        prologue=runner.prologue,
        epilogue=runner.epilogue,
        includes=runner.includes,
        parameters=runner.parameters,
        pass_config=pass_config,
    )
    data0 = relay.var("data", shape=ishape, dtype=dtype)

    weight0 = relay.var("weight0", shape=wshape, dtype=dtype)
    weight1 = relay.var("weight1", shape=wshape, dtype=dtype)

    out2 = relay.nn.conv2d(data0, weight0, kernel_size=(3, 3), padding=(1, 1), groups=groups)
    out2 = relay.nn.relu(out2)
    out2 = relay.nn.conv2d(out2, weight1, kernel_size=(3, 3), padding=(1, 1), groups=groups)
    out2 = relay.nn.relu(out2)

    main_f = relay.Function([data0, weight0, weight1], out2)
    
    mod = tvm.IRModule.from_expr(main_f)
    
    # Emit tir
    @tvm.tir.transform.prim_func_pass(opt_level=0)
    def dump_tir(f, mod, ctx):
        #print(f)
        return f
    with tvm.transform.PassContext(opt_level=3, config={"tir.add_lower_pass": [(3, dump_tir)]}):
        lib = tvm.relay.build(mod, target='llvm')

    print(main_f.astext())
    mod = tvm.IRModule()
    mod["main"] = main_f
    mod = transform.InferType()(mod)

    i_data = np.random.randn(*ishape).astype(dtype)
    w0_data = np.random.randn(*wshape).astype(dtype)
    w1_data = np.random.randn(*wshape).astype(dtype)

    inputs = OrderedDict([("data", i_data), ("weight0", w0_data), ("weight1", w1_data)])

    output_list = generate_ref_data(mod, inputs)
    return mod, inputs, output_list, runner

def main():
    mod, inputs, output_list, runner = create_conv2d()

    uma_backend = VanillaAcceleratorBackend()
    uma_backend.register()
    mod = uma_backend.partition(mod)
    target = tvm.target.Target("vanilla_accelerator", host=tvm.target.Target("c"))
    target_c = tvm.target.Target("c")
    
    export_directory = tvm.contrib.utils.tempdir(keep_for_debug=True).path
    print(f"Generated files are in {export_directory}")
    compile_and_run(
        AOTModel(module=mod, inputs=inputs, outputs=output_list),
        runner,
        interface_api="c",
        use_unpacked_api=True,
        target=[target_c, target],
        test_dir=str(export_directory),
    )

if __name__ == "__main__":
    main()
