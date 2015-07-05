CU_APPS=simple2DFD simpleMultiGPU simpleP2P_PingPong
C_APPS=simpleC2C simpleP2P simpleP2P_CUDA_Aware

all: ${C_APPS} ${CU_APPS}

simpleC2C: simpleC2C.c
	gcc -O2 -std=c99 -I${MPI_HOME}/include -L${MPI_HOME}/lib -lmpi -o simpleC2C simpleC2C.c
simpleP2P: simpleP2P.c
	gcc -O2 -std=c99 -I${MPI_HOME}/include -I${CUDA_HOME}/include -L${MPI_HOME}/lib -L${CUDA_HOME}/lib64 -lcudart -lmpi -o simpleP2P simpleP2P.c
simpleP2P_CUDA_Aware: simpleP2P_CUDA_Aware.c
	gcc -O2 -std=c99 -I${MPI_HOME}/include -I${CUDA_HOME}/include -L${MPI_HOME}/lib -L${CUDA_HOME}/lib64 -lcudart -lmpi -o simpleP2P_CUDA_Aware simpleP2P_CUDA_Aware.c
%: %.cu
	nvcc -O2 -arch=sm_20 -I${MPI_HOME}/include -o $@ $<
%: %.c
	gcc -O2 -std=c99 -I${MPI_HOME}/include -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
