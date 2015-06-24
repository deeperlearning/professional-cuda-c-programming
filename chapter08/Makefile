CU_APPS=cublas cuda-openacc cufft-multi cufft cusparse rand-kernel \
        replace-rand-streams replace-rand
C_APPS=simple-data simple-kernels simple-parallel

all: ${C_APPS} ${CU_APPS}

cublas: cublas.cu
	nvcc -O2 -arch=sm_20 -lcublas -o cublas cublas.cu
cuda-openacc: cuda-openacc.cu
	nvcc -O2 -arch=sm_20 -lcublas -lcurand -o cuda-openacc cuda-openacc.cu
cufft-multi: cufft-multi.cu
	nvcc -O2 -arch=sm_20 -lcufft -o cufft-multi cufft-multi.cu
cufft: cufft.cu
	nvcc -O2 -arch=sm_20 -lcufft -o cufft cufft.cu
cusparse: cusparse.cu
	nvcc -O2 -arch=sm_20 -lcusparse -o cusparse cusparse.cu
rand-kernel: rand-kernel.cu
	nvcc -O2 -arch=sm_20 -lcurand -o rand-kernel rand-kernel.cu
replace-rand-streams: replace-rand-streams.cu
	nvcc -O2 -arch=sm_20 -lcurand -o replace-rand-streams replace-rand-streams.cu
replace-rand: replace-rand.cu
	nvcc -O2 -arch=sm_20 -lcurand -o replace-rand replace-rand.cu
%: %.cu
	nvcc -O2 -arch=sm_20 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
