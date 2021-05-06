# matmul-conv-op-impl

## Setup
Open the outer `CMakeLists.txt` and you'll find things like
```cmake
set(build-cpu ON)
set(build-gpu ON)
set(build-mlu OFF)
```
You should adjust to your own need.

```shell script
# clone repo
git clone --recursive https://github.com/bulffi/matmul-conv-op-impl.git
# setup cpp side
./vcpkg/bootstrap-vcpkg.sh
./vcpkg/vcpkg install eigen3
# setup python side
conda create -n codesign_op python=3.8
conda activate codesign_op
pip install numpy
# use cmake
mkdir build
cd build
cmake ..
# cmake .. -DCMAKE_BUILD_TYPE=Release
make cpu_op
# make gpu_op
# make mlu_op
cp cpu/cpu_op.cpython-38-x86_64-linux-gnu.so ..
# or whatever name on your machine depending on the device you choose and
# the OS you are using
python test.py
```