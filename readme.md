# TVM UMA - Getting started with VSCODE
You need cmake and llvm for the following steps.

## Build & Installation
1. Build TVM:

    ```bash
    chmod +x build_tvm.sh
    ./build_tvm.sh
    ```

2. After successful build you should see the following output:
    ```bash
    "PYTHONPATH: $PYTHONPATH"
    "TVM_LIBRARY_PATH: $TVM_LIBRARY_PATH"
    "LD_LIBRARY_PATH: $LD_LIBRARY_PATH
    ```

3. Enable virtual environments in VSCode. Create a new environment using the `requirements.txt`.

4. Execute `python3 train_model.py` inside the environment.

5. Execute `run.py` inside the environment.
