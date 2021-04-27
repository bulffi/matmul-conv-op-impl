mlu.o: mlu_fn.o mlu_stub.o
	g++ -L${NEUWARE_HOME}/lib64 mlu_stub.o mlu_fn.o -lcnrt -o mlu.o


mlu_fn.o: mlu_op.mlu
	cncc -c --bang-mlu-arch=MLU200 mlu_op.mlu -o mlu_fn.o

mlu_stub.o:
	g++ -I${NEUWARE_HOME}/include -c mlu_op.cpp -std=c++11 -o mlu_stub.o
