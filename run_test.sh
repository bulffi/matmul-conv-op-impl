cd cmake-build-release
make cpu_op
cp CPU/cpu_op.cpython-38-x86_64-linux-gnu.so ../
cd ..

cd cmake-build-release
make gpu_op
cp GPU/gpu_op.cpython-38-x86_64-linux-gnu.so ../
cd ..

#make mlu_op
#cp MLU/mlu_op.so ../
#cd ..

python3 test.py