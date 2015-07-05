CU_APPS=cusparse-matrix-matrix access-ordering cublas-matrix-matrix \
		cublas-matrix-matrix-async cufft-inverse cusparse-matrix-matrix-double
C_APPS=

all: ${C_APPS} ${CU_APPS}

cusparse-matrix-matrix: cusparse-matrix-matrix.cu
	nvcc -O2 -arch=sm_20 -o cusparse-matrix-matrix cusparse-matrix-matrix.cu -lcusparse

cublas-matrix-matrix: cublas-matrix-matrix.cu
	nvcc -O2 -arch=sm_20 -o cublas-matrix-matrix cublas-matrix-matrix.cu -lcublas

cublas-matrix-matrix-async: cublas-matrix-matrix-async.cu
	nvcc -O2 -arch=sm_20 -o cublas-matrix-matrix-async cublas-matrix-matrix-async.cu -lcublas

cufft-inverse: cufft-inverse.cu
	nvcc -O2 -arch=sm_20 -o cufft-inverse cufft-inverse.cu -lcufft -lm

cusparse-matrix-matrix-double: cusparse-matrix-matrix-double.cu
	nvcc -O2 -arch=sm_20 -o cusparse-matrix-matrix-double cusparse-matrix-matrix-double.cu -lcusparse

%: %.cu
	nvcc -O2 -arch=sm_20 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -lm -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
