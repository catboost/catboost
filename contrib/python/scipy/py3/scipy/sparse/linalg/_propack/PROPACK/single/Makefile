#
#     (C) Rasmus Munk Larsen, Stanford University, 2004
#
include ../make.inc
IPATH = -I.

all: lib exampleprogs

### Object files.

DOBJ = slanbpro.o sreorth.o sgetu0.o ssafescal.o sblasext.o \
      slansvd.o printstat.o sgemm_ovwr.o  \
      slansvd_irl.o sbsvd.o sritzvec.o s$(MGS).o second.o 

.f.o:
	$(F77) $(FFLAGS) $(IPATH) -c -o $*.o $*.f
.F.o:
	$(F77) $(FFLAGS) $(IPATH) -c -o $*.o $*.F
.c.o:
	$(CC)  $(CFLAGS) $(IPATH) -c -o $*.o $*.c


lib: liblapack libspropack_$(PLAT).a 

libspropack_$(PLAT).a: $(DOBJ)
	rm -f libspropack_$(PLAT).a
	$(MAKELIB) libspropack_$(PLAT).a $(DOBJ) 
	$(RANLIB)  libspropack_$(PLAT).a

liblapack:
	@( cd Lapack_Util; \
	$(MAKE) lib; \
	cd .. )		

exampleprogs: 
	@( cd Examples; \
	$(MAKE) all; \
	cd .. )

test: 
	@( cd Examples; \
	$(MAKE) test; \
	cd .. )

verify: 
	@( cd Examples; \
	$(MAKE) verify; \
	cd .. )


clean:	
	rm -f  *.o *.il  lib*_$(PLAT).a
	rm -rf rii_files
	@( cd Examples; \
	$(MAKE) clean; \
	cd ..; \
	cd Lapack_Util; \
	$(MAKE) clean; \
	cd .. )

cleanall:	
	rm -f  *.o  *.a *~ *.il
	rm -rf rii_files
	@( cd Examples; \
	$(MAKE) cleanall; \
	cd ..; \
	cd Lapack_Util; \
	$(MAKE) cleanall; \
	cd .. )

