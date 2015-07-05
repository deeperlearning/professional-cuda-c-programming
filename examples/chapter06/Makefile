CU_APPS=asyncAPI simpleCallback simpleHyperqBreadth simpleHyperqDependence \
        simpleHyperqDepth simpleHyperqOpenmp simpleMultiAddBreadth \
        simpleMultiAddDepth
C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_20 -Xcompiler -fopenmp -o $@ $< -lgomp
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
