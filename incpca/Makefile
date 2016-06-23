
HOME=/home/master/04/weitang114

CFLAGS=-I$(HOME)/magma/include -O3 -g #-DDEBUGMODE 
LDFLAGS=-L$(HOME)/magma/lib -lmagma -lcuda -lcublas

inc_pca: inc_pca.cu
	nvcc inc_pca.cu -o $@ $(CFLAGS) $(LDFLAGS)
	

testevd:
	nvcc -I$(HOME)/magma/include -L$(HOME)/magma/lib -lmagma -c test_evd.cu -o test_evd.o -g
	nvcc test_evd.o -L$(HOME)/magma/lib -lmagma -lcuda -lcublas -o test_evd

clean:
	rm inc_pca
