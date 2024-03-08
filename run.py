from tvm.micro.testing.aot_test_utils import AOT_DEFAULT_RUNNER
import tvm
from tvm import relay
from tvm.relay import transform
from collections import OrderedDict
import numpy as np
import tensorflow as tf


from tvm.testing.aot import (
    AOTTestModel as AOTModel,
    AOTTestRunner as AOTRunner,
    generate_ref_data,
    compile_and_run,
)

# Load pretrained CNN
model = tf.keras.models.load_model('mnist_cnn.h5')

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

# Create TVM relay representation
runner = AOTRunner(
    makefile=AOT_DEFAULT_RUNNER.makefile,
    prologue=AOT_DEFAULT_RUNNER.prologue,
    epilogue=AOT_DEFAULT_RUNNER.epilogue,
    includes=AOT_DEFAULT_RUNNER.includes,
    parameters=AOT_DEFAULT_RUNNER.parameters
)

input_placeholder   = relay.var("input",   shape=input_shape,         dtype=dtype)
weight0_placeholder = relay.var("weight0", shape=weigths_conv0_shape, dtype=dtype)
weight1_placeholder = relay.var("weight1", shape=weigths_conv1_shape, dtype=dtype)
dense_placeholder   = relay.var("dense",   shape=weigths_dense_shape, dtype=dtype)

out = relay.nn.conv2d(input_placeholder, weight0_placeholder, padding=(1, 1), data_layout='NHWC', kernel_layout='HWIO')
out = relay.nn.relu(out)
out = relay.nn.conv2d(out, weight1_placeholder, kernel_size=(3, 3), padding=(1, 1), data_layout='NHWC', kernel_layout='HWIO')
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


target_c = tvm.target.Target("c")
export_directory = "uma_output/files"
print(f"Generated files are in {export_directory}")
compile_and_run(
    AOTModel(module=mod, inputs=inputs, outputs=output_list),
    runner,
    interface_api="c",
    use_unpacked_api=True,
    target=target_c,
    test_dir=str(export_directory),
)
