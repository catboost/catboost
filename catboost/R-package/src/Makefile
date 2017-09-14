unexport YA_BUILD_RESULTS_REPORT

.PHONY : all

THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

all:
	chmod +rx $(THIS_DIR)/../../../ya
	$(THIS_DIR)/../../../ya make -r
