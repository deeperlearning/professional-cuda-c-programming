CU_APPS=globalVariable1 globalVariable2 memTransfer pinMemTransfer \
		readWriteSegment readWriteSegmentUnroll transpose sumArrayZerocpyUVA \
		sumArrayZerocpy-offset sumArrayZerocpyUVA-offset simpleMathAoS-align \
		simpleMathAoS-x
C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_20 -o $@ $<
%: %.c
	g++ -O2 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
