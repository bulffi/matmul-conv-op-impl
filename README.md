# matmul-conv-op-impl

## Setup
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
```