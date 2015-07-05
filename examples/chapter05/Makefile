CU_APPS=checkSmemRectangle checkSmemSquare constantReadOnly constantStencil \
        reduceInteger reduceIntegerShfl simpleShfl transposeRectangle
C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_30 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
