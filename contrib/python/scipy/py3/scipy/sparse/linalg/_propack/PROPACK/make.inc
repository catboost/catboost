###  Compiler flags for fortran and C modules.

#  Sun
PLAT = SUNOS

F77 = f90  
FFLAGS = -O3 -pg 
FFLAGS_NOOPT = -O0 -pg 
FFLAGS_ACCURATE = $(FFLAGS) 
CC = cc
CFLAGS = -O3
LINKER = f90
MAKELIB = ar rc
LINKFLAGS = -O3 -pg 
BLAS =  -xlic_lib=sunperf
RANLIB = ranlib
MGS = mgs.risc
