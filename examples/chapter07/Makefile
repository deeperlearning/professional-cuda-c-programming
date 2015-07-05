CU_APPS=atomic-ordering floating-point-accuracy floating-point-perf fmad \
        intrinsic-standard-comp my-atomic-add nbody
C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_20 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
