CU_APPS=reduceInteger-16 reduceInteger-sync reduceFloatGpu nestedHelloWorldNew \
		nestedHelloWorldLimited reduceInteger-8-new
C_APPS=reduceFloat

all: ${C_APPS} ${CU_APPS}

nestedHelloWorldNew: nestedHelloWorldNew.cu
	nvcc -O2 -arch=sm_35 -rdc=true -o nestedHelloWorldNew \
				   nestedHelloWorldNew.cu -lcudadevrt
nestedHelloWorldLimited: nestedHelloWorldLimited.cu
	nvcc -O2 -arch=sm_35 -rdc=true -o nestedHelloWorldLimited \
				   nestedHelloWorldLimited.cu -lcudadevrt

%: %.cu
	nvcc -O2 -arch=sm_20 -o $@ $<
%: %.c
	g++ -O2 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
