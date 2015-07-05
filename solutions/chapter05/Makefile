CU_APPS=checkSmemSquare checkSmemRectangle constantReadOnlyGlobal \
		reduceInteger simpleShflAltered
C_APPS=

all: ${C_APPS} ${CU_APPS}

simpleShflAltered: simpleShflAltered.cu
	nvcc -O2 -arch=sm_30 -o $@ $<

%: %.cu
	nvcc -O2 -arch=sm_20 -o $@ $<
%: %.c
	g++ -O2 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
