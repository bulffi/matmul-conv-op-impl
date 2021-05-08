libmlu.so: mlu_fn.o mlu_stub.o
	g++ -L${NEUWARE_HOME}/lib64 mlu_stub.o mlu_fn.o -lcnrt -shared -o libmlu.so


mlu_fn.o: mlu_op.mlu
	cncc --bang-mlu-arch=MLU200 mlu_op.mlu -o mlu_fn.o

mlu_stub.o: mlu_op.cpp
	g++ -I${NEUWARE_HOME}/include -std=c++11 -c -fPIC mlu_op.cpp -o mlu_stub.o

clean:
	rm *.o
	rm *.so
