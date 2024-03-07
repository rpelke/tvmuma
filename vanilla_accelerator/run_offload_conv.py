from tvm.micro.testing.aot_test_utils import AOT_DEFAULT_RUNNER
import tvm
from tvm import relay
from tvm.relay import transform
from collections import OrderedDict
from backend import VanillaAcceleratorBackend
import numpy as np
import tensorflow as tf
import os


from tvm.testing.aot import (
    AOTTestModel as AOTModel,
    AOTTestRunner as AOTRunner,
    generate_ref_data,
    compile_and_run,
)


def create_conv2d(groups=1, runner=AOT_DEFAULT_RUNNER, weight_shape=32):
    # Load pretrained CNN
    model = tf.keras.models.load_model(os.path.abspath('test_niko/mnist_cnn.h5'))

    # Extract parameters from TF CNN
    dtype = "float32"
    input_shape = (1, 28, 28, 1)
    weigths_conv0 = model.layers[0].get_weights()[0]
    weigths_conv1 = model.layers[1].get_weights()[0]
    weigths_dense = model.layers[3].get_weights()[0]
    weigths_dense = np.swapaxes(weigths_dense, 0, 1)

    # Shapes
    weigths_conv0_shape = model.layers[0].get_weights()[0].shape
    weigths_conv1_shape = model.layers[1].get_weights()[0].shape
    weigths_dense_shape = weigths_dense.shape

    pass_config = {"tir.usmp.enable": True}
    runner = AOTRunner(
        makefile=runner.makefile,
        prologue=runner.prologue,
        epilogue=runner.epilogue,
        includes=runner.includes,
        parameters=runner.parameters,
        pass_config=pass_config,
    )
    
    input_placeholder   = relay.var("input",   shape=input_shape,         dtype=dtype)
    weight0_placeholder = relay.var("weight0", shape=weigths_conv0_shape, dtype=dtype)
    weight1_placeholder = relay.var("weight1", shape=weigths_conv1_shape, dtype=dtype)
    dense_placeholder   = relay.var("dense",   shape=weigths_dense_shape, dtype=dtype)

    # Our acclerator need NCHW layout, so we need to transform the layout.
    tr_i = relay.layout_transform(input_placeholder, src_layout="NHWC", dst_layout="NCHW")
    tr_w = relay.layout_transform(weight0_placeholder, src_layout="HWIO", dst_layout="OIHW")
    out = relay.nn.conv2d(tr_i, tr_w, kernel_size=(3, 3), padding=(1, 1), data_layout='NCHW', kernel_layout='OIHW', groups=1)
    out = relay.nn.relu(out)
    tr_w = relay.layout_transform(weight1_placeholder, src_layout="HWIO", dst_layout="OIHW")
    out = relay.nn.conv2d(out, tr_w, kernel_size=(3, 3), padding=(1, 1), data_layout='NCHW', kernel_layout='OIHW', groups=1)
    out = relay.nn.relu(out)
    out = relay.nn.batch_flatten(out)
    out = relay.nn.dense(out, dense_placeholder)
    out = relay.nn.softmax(out)
    
    main_f = relay.Function([input_placeholder, weight0_placeholder, weight1_placeholder, dense_placeholder], out)
    
    mod = tvm.IRModule()
    mod["main"] = main_f
    mod = transform.InferType()(mod)
    print(main_f)

   # Load the dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255
    x_test = x_test[0,:,:]
    x_test = np.expand_dims(x_test, -1)
    x_test = np.expand_dims(x_test, 0)

    inputs = OrderedDict([("input", x_test), ("weight0", weigths_conv0), ("weight1", weigths_conv1), ("dense", weigths_dense)])
    output_list = generate_ref_data(mod, inputs)

    inputs = OrderedDict([("input", x_test), ("weight0", weigths_conv0), ("weight1", weigths_conv1), ("dense", weigths_dense)])

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
