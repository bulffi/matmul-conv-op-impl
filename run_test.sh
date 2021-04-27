cd cmake-build-debug

make cpu_op
cp CPU/cpu_op.cpython-38-x86_64-linux-gnu.so ../

# make gpu_op
# cp GPU/gpu_op.cpython-38-x86_64-linux-gnu.so ../

cd ..
python3 test.py