unexport YA_BUILD_RESULTS_REPORT

.PHONY : all

THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

all:
	$(THIS_DIR)/../../../ya make -r
