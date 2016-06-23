CUDALAGS=-std=c++11 -ccbin=/usr/bin/g++-4.9 -arch=sm_35 -lcublas -lcusolver
CPUFLAGS=-llapacke -lcblas

all:
	nvcc $(CUDAFLAGS) sgdpca.cu 
	g++  $(CPUFLAGS) sgdpca.cpp

cuda:
	nvcc $(CUDAFLAGS) sgdpca.cu

cpu:
	g++ $(CPUFLAGS) sgdpca.cpp

clean:
	rm a.out
