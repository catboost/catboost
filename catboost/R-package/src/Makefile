unexport YA_BUILD_RESULTS_REPORT
unexport LD_LIBRARY_PATH
unexport DYLD_LIBRARY_PATH

.PHONY : all

THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

all:
	chmod +rx $(THIS_DIR)/../../../ya
	$(THIS_DIR)/../../../ya make -r
