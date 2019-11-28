CUDA_PATH ?= /usr/local/cuda-10.1
NVCC := $(CUDA_PATH)/bin/nvcc
INCLUDES_CUDA :=$(CUDA_PATH)/include
LIBDIR_CUDA :=$(CUDA_PATH)/lib64
INCLUDES_CUDNN := /usr/local/cuda-10.1/include
LIBDIR_CUDNN := /usr/local/cuda-10.1/lib64



CSRCS := $(shell find . -name '*.cpp' -not -name '._*')
COBJS := $(subst .cpp,.o,$(CSRCS))

CUSRCS := $(shell find . -name '*.cu' -not -name '._*')
CUOBJS := $(subst .cu,.o,$(CUSRCS))

CUFLAGS= \
-I. \
-Xcompiler \
-fPIC \
-std=c++11

LDFLAGS=-L. -lm  -lrt

FLAGS = \
-I ./include \
-I$(INCLUDES_CUDNN) \
-L$(LIBDIR_CUDA) -lcudnn \
-std=c++11


all: run


run: reduction.cu
	$(NVCC) -o run reduction.cu -arch=sm_70 -L$(LIBDIR_CUDA) -lcudart -lcuda -lcublas -lcudnn

clean:
	#find . -name "*.o" -exec rm -f '{}' ';'
	#rm -f run
