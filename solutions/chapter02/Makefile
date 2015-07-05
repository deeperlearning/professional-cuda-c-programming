CU_APPS=sumMatrixOnGPU-2D-grid-2D-block-integer \
		sumMatrixOnGPU-2D-grid-1D-block-two
C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_20 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
