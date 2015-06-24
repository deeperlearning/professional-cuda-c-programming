CU_APPS=crypt.config crypt.constant crypt.flexible crypt.legacy crypt.openmp \
        crypt.overlap crypt.parallelized debug-hazards debug-segfault \
        debug-segfault.fixed sumMatrixGPU sumMatrixGPU_nvToolsExt
C_APPS=crypt generate_data generate_userkey

all: ${C_APPS} ${CU_APPS}

crypt.openmp: crypt.openmp.cu
	nvcc -Xcompiler -fopenmp -O2 -arch=sm_20 -o crypt.openmp crypt.openmp.cu -lgomp
sumMatrixGPU_nvToolsExt: sumMatrixGPU_nvToolsExt.cu
	nvcc -O2 -arch=sm_20 -L${CUDA_HOME}/lib64 -lnvToolsExt -o sumMatrixGPU_nvToolsExt sumMatrixGPU_nvToolsExt.cu
%: %.cu
	nvcc -O2 -arch=sm_20 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
