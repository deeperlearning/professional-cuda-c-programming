CU_APPS=checkDeviceInfor checkThreadIndex sumArraysOnGPU-timer \
        sumMatrixOnGPU-1D-grid-1D-block sumMatrixOnGPU-2D-grid-2D-block \
        checkDimension defineGridBlock sumArraysOnGPU-small-case \
        sumMatrixOnGPU-2D-grid-1D-block sumMatrixOnGPU
C_APPS=sumArraysOnHost

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_20 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
