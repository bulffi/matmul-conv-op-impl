libmlu.so: mlu_fn_base.o mlu_stub.o mlu_fn_ram.o mlu_fn_multi.o mul_fn_multi_sram.o
	g++ -L${NEUWARE_HOME}/lib64 -O3 mlu_stub.o mlu_fn_base.o mlu_fn_ram.o mlu_fn_multi.o mul_fn_multi_sram.o -lcnrt -shared -o libmlu.so

mlu_fn_base.o: mlu_op_base.mlu
	cncc --bang-mlu-arch=MLU200 -O3 mlu_op_base.mlu -o mlu_fn_base.o

mlu_fn_ram.o: mlu_op_ram.mlu
	cncc --bang-mlu-arch=MLU200 -O3  mlu_op_ram.mlu -o mlu_fn_ram.o

mlu_fn_multi.o: mlu_op_multi_core.mlu
	cncc --bang-mlu-arch=MLU200 -O3 mlu_op_multi_core.mlu -o mlu_fn_multi.o

mul_fn_multi_sram.o: mlu_op_multi_core_sram.mlu
	cncc --bang-mlu-arch=MLU200 -O3 mlu_op_multi_core_sram.mlu -o mul_fn_multi_sram.o

mlu_stub.o: mlu_op.cpp
	g++ -I${NEUWARE_HOME}/include -O3 -std=c++11 -c -fPIC mlu_op.cpp -o mlu_stub.o

clean:
	rm *.o
	rm *.so
