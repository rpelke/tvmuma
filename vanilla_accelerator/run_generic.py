from tvm.micro.testing.aot_test_utils import AOT_DEFAULT_RUNNER
import tvm
from tvm import relay
from tvm.relay import transform
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from backend import VanillaAcceleratorBackend 
import numpy as np

from tvm.testing.aot import (
    AOTTestModel as AOTModel,
    AOTTestRunner as AOTRunner,
    generate_ref_data,
    compile_and_run,
)

def parametersToC(tvmgen_default_weights_size, tvmgen_default_dense_size, TVMGEN_DEFAULT_INPUT_SIZE, TVMGEN_DEFAULT_OUTPUT_SIZE):
    # Öffnen der Datei "output.h" im Schreibmodus
    with open("vanilla_accelerator/files/neededParameters.h", "w") as file:
        # Schreiben des C++-Header-Datei-Präambels
        file.write("#ifndef OUTPUT_H\n")
        file.write("#define OUTPUT_H\n\n")

        # Schreiben der TVM_DEFAULT_WEIGHTS_SIZE-Definition
        file.write("const int TVM_DEFAULT_WEIGHTS_SIZE[] = {\n")
        for value in tvmgen_default_weights_size:
            file.write(f"    {value},\n")
        file.write("};\n\n")

        # Schreiben der TVM_DEFAULT_DENSE_SIZE-Definition
        file.write("const int TVM_DEFAULT_DENSE_SIZE[] = {\n")
        for value in tvmgen_default_dense_size:
            file.write(f"    {value},\n")
        file.write("};\n\n")

        # Schreiben der TVM_DEFAULT_INPUT_SIZE-Definition
        file.write(f"const unsigned int TVM_DEFAULT_INPUT_SIZE = {TVMGEN_DEFAULT_INPUT_SIZE};\n\n")

        # Schreiben der TVM_DEFAULT_OUTPUT_SIZE-Definition
        file.write(f"const unsigned int TVM_DEFAULT_OUTPUT_SIZE = {TVMGEN_DEFAULT_OUTPUT_SIZE};\n\n")
        
        # Schließen des Header-Datei-Codes
        file.write("#endif // OUTPUT_H\n")

def setStruct(w_iterator, d_iterator):
    header_code = "#ifndef SET_STRUCT_H\n"
    header_code += "#define SET_STRUCT_H\n\n"
    
    header_code += "#include <tvmgen_default.h>\n\n"
    header_code += "#include \"tvmgen_default_input_data_input.h\"\n\n"
    for i in range(w_iterator):
        header_code += "#include \"tvmgen_default_input_data_weight" + str(i) + ".h\"\n\n"
        
    for i in range(d_iterator):
       header_code += "#include \"tvmgen_default_input_data_dense" + str(i) + ".h\"\n\n"
    
    header_code += "void setStruct(struct tvmgen_default_inputs* input_struct) {\n"
    header_code += "    input_struct->input = tvmgen_default_input_data_input;\n"
    for i in range(w_iterator):
        header_code += "    input_struct->weight" + str(i) + " = tvmgen_default_input_data_weight" + str(i) + ";\n"
    
    for i in range(d_iterator):
        header_code += "    input_struct->dense" + str(i) + " = tvmgen_default_input_data_weight" + str(i) + ";\n"
    header_code += "}\n\n"
    
    header_code += "#endif // SET_STRUCT_H\n"
    
    with open("vanilla_accelerator/files/setStruct.h", "w") as header_file:
        header_file.write(header_code)
        
def tvmgenMainCall(w_iterator, d_iterator):
    header_code = "#ifndef MAIN_CALL_H\n"
    header_code += "#define MAIN_CALL_H\n\n"
    
    header_code += "#include <tvmgen_default.h>\n"
    header_code += "#include <stdint.h>\n\n"
        
    header_code += "void callTVMmain(struct tvmgen_default_inputs *tvmgen_default_inputs, struct tvmgen_default_outputs *tvmgen_default_outputs, uint8_t* workspace) {\n"
    header_code += "    tvmgen_default___tvm_main__(\n"
    header_code += "        tvmgen_default_inputs->input,\n"
    for i in range(w_iterator):
        header_code += "        tvmgen_default_inputs->weight" + str(i) + ",\n"
    
    for i in range(d_iterator):
        header_code += "        tvmgen_default_inputs->dense" + str(i) + ",\n"
    header_code += "        tvmgen_default_outputs->output,\n"
    header_code += "        workspace);\n"
    header_code += "}\n\n"
    
    header_code += "#endif // MAIN_CALL_H\n"

    with open("vanilla_accelerator/files/tvmgenMainCall.h", "w") as header_file:
        
        header_file.write(header_code)
        
def setPointer(w_iterator, d_iterator):
    header_code = "#ifndef SET_POINTER_H\n"
    header_code += "#define SET_POINTER_H\n\n"
    
    header_code += "#include <tvmgen_default.h>\n\n"
    
    header_code += "void setPointer(struct tvmgen_default_inputs* input_struct, void* shmp_start, const int TVM_DEFAULT_INPUT_SIZE, const int TVM_DEFAULT_WEIGHTS_SIZE[], const int TVM_DEFAULT_DENSE_SIZE[]) {\n"
    header_code += "    input_struct->input = shmp_start;\n"
    for i in range(w_iterator):
        if i == 0:
            header_code += "    input_struct->weight0 = input_struct->input  + TVM_DEFAULT_INPUT_SIZE;\n"
            continue
        header_code += "    input_struct->weight" + str(i) + " = input_struct->weight" + str(i-1) + " + TVM_DEFAULT_WEIGHTS_SIZE[" + str(i-1) + "];\n"
    
    for i in range(d_iterator):
        if i == 0:    
            header_code += "    input_struct->dense" + str(i) + " = input_struct->weight" + str(w_iterator-1) + " + TVM_DEFAULT_WEIGHTS_SIZE[" + str(w_iterator-1) + "];\n"
            continue
        header_code += "    input_struct->dense" + str(i) + " = input_struct->dense" + str(i-1) + " + TVM_DEFAULT_DENSE_SIZE[" + str(i-1) + "];\n"
    header_code += "}\n\n"
    
    header_code += "#endif // SET_POINTER_H\n"
    
    with open("vanilla_accelerator/files/setPointer.h", "w") as header_file:
        header_file.write(header_code)

    # Load pretrained CNN
def modelStructure():
    
    pass_config = {"tir.usmp.enable": True}
    # Create TVM relay representation
    runner = AOTRunner(
        makefile=AOT_DEFAULT_RUNNER.makefile,
        prologue=AOT_DEFAULT_RUNNER.prologue,
        epilogue=AOT_DEFAULT_RUNNER.epilogue,
        includes=AOT_DEFAULT_RUNNER.includes,
        parameters=AOT_DEFAULT_RUNNER.parameters,
        pass_config=pass_config,
    )
    
    #modelPath = input("filepath of the model: ")
    model = tf.keras.models.load_model("test_niko/flo_mnist_cnn2.h5") #modelPath)

    #test_niko/mnist_cnn.h5
    #test_niko/flo_mnist_cnn.h5
    #test_niko/flo_mnist_cnn2.h5

    dtype = "float32"

    conv_weights = {}
    weight_shapes = {}
    weights_placeholder = []

    mp_weights = []

    weigths_dense = []#None
    dense_placeholder = []#None

    input_shape = model.layers[0].input_shape
    input_shape = (1,) + input_shape[1:len(input_shape)]

    output_shape = model.layers[-1].output_shape
    output_shape = (1,) + output_shape[1:len(output_shape)]

    input_placeholder   = relay.var("input", shape=input_shape, dtype=dtype)
    
    w_iterator = 0
    d_iterator = 0
    
    out = input_placeholder
    out = relay.layout_transform(out, src_layout="NHWC", dst_layout="NCHW")

    # Extract parameters from TF CNN
    for l in model.layers : #l = layers
        if isinstance(l, tf.keras.layers.Conv2D) :
            conv_weights[l.name] = l.get_weights()[0]
            weight_shapes[l.name] = l.get_weights()[0].shape
            weights_placeholder.append(relay.var("weight" + str(w_iterator) , shape=weight_shapes[l.name], dtype=dtype))
            
            tr_w = relay.layout_transform(weights_placeholder[w_iterator], src_layout="HWIO", dst_layout="OIHW")
            
            """ if iterator == 0:
                kernel_size = None
            else:
                kernel_size = l.kernel_size
                """
                
            if l.strides != (1,1) :
                raise Exception("This case is not implemented.")
            
            if l.padding == 'same' :
                padding = (l.kernel_size[0] // 2, l.kernel_size[1] // 2)
            elif l.padding == 'valid':
                padding = (0,0) 
            else:
                raise Exception("incompatible padding value!")
            
            if l.use_bias :
                raise Exception("Bias not implemented.")
                
            # channels
            print("input_shape: {}".format(l.input_shape))
            print("kernel.shape: {}".format(l.kernel.shape))
            print("weight.shape: {}".format(weight_shapes[l.name]))
            print("output_shape: {}\n".format(l.output_shape))
            
            out = relay.nn.conv2d(out, tr_w,  padding=padding, kernel_size= l.kernel_size, data_layout='NCHW', kernel_layout='OIHW', groups=1)
            
            
            if l.activation.__name__ == 'relu':
                out = relay.nn.relu(out)
            elif l.activation.__name__ == 'softmax':
                out = relay.nn.softmax(out)
                
            w_iterator = w_iterator + 1
            
        elif isinstance(l, tf.keras.layers.MaxPool2D):
            print("input_shape: {}".format(l.input_shape))
            print("output_shape: {}\n".format(l.output_shape))
            
            out = relay.nn.max_pool2d(out, pool_size=l.pool_size, strides=l.strides, layout='NCHW', out_layout='NCHW')
        
        elif isinstance(l, tf.keras.layers.Flatten):
            out = relay.layout_transform(out, src_layout="NCHW", dst_layout="NHWC")
            out = relay.nn.batch_flatten(out)
            
        elif isinstance(l, tf.keras.layers.Dense):
            weigths_dense.append(l.get_weights()[0])
            weigths_dense[d_iterator] = np.swapaxes(weigths_dense[d_iterator], 0, 1)
            weigths_dense_shape = weigths_dense[d_iterator].shape
            dense_placeholder.append(relay.var("dense" + str(d_iterator), shape=weigths_dense_shape, dtype=dtype))
            
            out = relay.nn.dense(out, dense_placeholder[d_iterator])
            if l.activation.__name__ == 'relu':
                out = relay.nn.relu(out)
            elif l.activation.__name__ == 'softmax':
                out = relay.nn.softmax(out)
                
            if l.use_bias :
                raise Exception("Bias not implemented.")
            
            d_iterator = d_iterator + 1
            
        else :
            raise Exception("Unknown layer type!")
        
    main_f = relay.Function([input_placeholder, *weights_placeholder, *dense_placeholder], out)
    

    mod = tvm.IRModule()
    mod["main"] = main_f
    mod = transform.InferType()(mod)
    print(main_f)

    #write the important data for shared meomry allocatin in a file
    TVMGEN_DEFAULT_INPUT_SIZE = 4*np.prod(np.array(input_shape))
    tvmgen_default_weights_size = []
    tvmgen_default_dense_size = []
    TVMGEN_DEFAULT_OUTPUT_SIZE = 4*np.prod(np.array(output_shape))

    weight_shapes_list = [i for i in weight_shapes.values()]

    for i in range(w_iterator):
        tvmgen_default_weights_size.append(np.prod(weight_shapes_list[i])*4)

    for i in range(d_iterator):
        tvmgen_default_dense_size.append(np.prod(weigths_dense[i].shape)*4)
        
    parametersToC(tvmgen_default_weights_size, tvmgen_default_dense_size, TVMGEN_DEFAULT_INPUT_SIZE, TVMGEN_DEFAULT_OUTPUT_SIZE)
    tvmgenMainCall(w_iterator, d_iterator)
    setStruct(w_iterator, d_iterator)
    setPointer(w_iterator, d_iterator)

    # Load the dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255
    x_test = x_test[0,:,:]
    x_test = np.expand_dims(x_test, -1)
    x_test = np.expand_dims(x_test, 0)

    index1 = 0
    inputList = [("input", x_test)]
    k = []
    for i in range(len(model.layers)):
        if isinstance(model.layers[i], tf.keras.layers.Conv2D):
            inputList = inputList + [("weight" + str(index1), conv_weights[model.layers[i].name])]
            k.append(conv_weights[model.layers[i].name])
            index1 = index1 + 1
            
    for i in range(len(weigths_dense)): 
        inputList = inputList + [("dense" + str(i), weigths_dense[i])]
        k.append(weigths_dense[i])

    ### Check Relay model ##########################################################
    from tvm.relay import create_executor
    executor = create_executor(kind="graph", mod=mod, target="llvm")
    res_relay = executor.evaluate()(x_test, *k)
    print(res_relay)
    
    res_tf = model(x_test)
    print(res_tf)
    ################################################################################
    
    inputs = OrderedDict(inputList)
    output_list = generate_ref_data(mod, inputs)
    
    return mod, inputs, output_list, runner

def main():
    mod, inputs, output_list, runner = modelStructure()
    
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
