unexport YA_BUILD_RESULTS_REPORT
unexport LD_LIBRARY_PATH
unexport DYLD_LIBRARY_PATH

.PHONY : all

THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
CUDA_ROOT := $(or $(CUDA_ROOT),$(wildcard /usr/local/cuda))
HAVE_CUDA := $(or $(HAVE_CUDA),$(if $(CUDA_ROOT),yes,no))

all:
	chmod +rx $(THIS_DIR)/../../../ya
	$(THIS_DIR)/../../../ya make -r -DHAVE_CUDA=$(HAVE_CUDA) -DCUDA_ROOT=$(CUDA_ROOT) -DCUDA_HOST_COMPILER=
