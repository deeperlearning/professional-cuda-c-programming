CU_APPS=globalVariable memTransfer pinMemTransfer readSegment \
		readSegmentUnroll simpleMathAoS simpleMathSoA sumArrayZerocpy \
		sumMatrixGPUManaged sumMatrixGPUManual transpose writeSegment
C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_20 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
