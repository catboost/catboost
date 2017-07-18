unexport YA_BUILD_RESULTS_REPORT

.PHONY : libcatboostr.so

THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

libcatboostr.so:
	$(THIS_DIR)/../../../ya make -r
