CU_APPS=nbody_intrinsic nbody_reduction
C_APPS=closestFP

all: ${C_APPS} ${CU_APPS}

nbody_reduction: nbody_reduction.cu
	nvcc -O2 -arch=sm_30 -o $@ $<

%: %.cu
	nvcc -O2 -arch=sm_20 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -lm -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
