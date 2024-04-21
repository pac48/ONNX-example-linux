# ONNX example linux
This repository contains a minimal example of running an ONNX model on Ubuntu 22.04.
The program accepts a file path to an image file and runs it through a simple model trained the MNIST dataset.


### Build the executable
Change directory into the root of this repository.
Download the latest onnxruntime release for linux.
```
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-1.17.3.tgz
```
Extract the contents into the root of the repository.
```
tar -xvzf onnxruntime-linux-x64-1.17.3.tgz
```
Create a build directory and build the executable. 
```
mkdir build
cd build
cmake ..
make
cd ..
```

### Run the model
The executable accepts a file path to an image to run the ONNX model on. 
The predicted number 0-9 will be printed. For example, run the following:
```
./build/onnx_mnist numbers/8.png
```
Note: the executable expects the `mnist.onnx` to be in the local directory.

