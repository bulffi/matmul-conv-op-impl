libmlu.so: mlu_fn_base.o mlu_stub.o mlu_fn_ram.o
	g++ -L${NEUWARE_HOME}/lib64 mlu_stub.o mlu_fn_base.o mlu_fn_ram.o -lcnrt -shared -o libmlu.so


mlu_fn_base.o: mlu_op_base.mlu
	cncc --bang-mlu-arch=MLU200 -g mlu_op_base.mlu -o mlu_fn_base.o

mlu_fn_ram.o: mlu_op_ram.mlu
	cncc --bang-mlu-arch=MLU200 -g mlu_op_ram.mlu -o mlu_fn_ram.o

mlu_stub.o: mlu_op.cpp
	g++ -I${NEUWARE_HOME}/include -g -std=c++11 -c -fPIC mlu_op.cpp -o mlu_stub.o

clean:
	rm *.o
	rm *.so
