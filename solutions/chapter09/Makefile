CU_APPS=simpleMultiGPUEvents simpleP2P_PingPong simpleP2P_PingPongDefault \
		simple2DFDModified simpleMultiGPUEvents-initial
C_APPS=simpleP2P_Pageable simpleP2P-async

all: ${C_APPS} ${CU_APPS}

simpleP2P_Pageable: simpleP2P_Pageable.c
	gcc -O2 -std=c99 -I${MPI_HOME}/include -I${CUDA_HOME}/include -L${MPI_HOME}/lib -L${CUDA_HOME}/lib64 -lcudart -lmpi -o simpleP2P_Pageable simpleP2P_Pageable.c

simpleP2P-async: simpleP2P-async.c
	gcc -O2 -std=c99 -I${MPI_HOME}/include -I${CUDA_HOME}/include -L${MPI_HOME}/lib -L${CUDA_HOME}/lib64 -lcudart -lmpi -o simpleP2P-async simpleP2P-async.c

%: %.cu
	nvcc -O2 -arch=sm_20 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -lm -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
