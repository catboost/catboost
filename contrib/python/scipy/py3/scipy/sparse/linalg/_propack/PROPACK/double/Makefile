#
#     (C) Rasmus Munk Larsen, Stanford University, 2004
#
include ../make.inc
IPATH = -I.

all: lib exampleprogs

### Object files.

DOBJ = dlanbpro.o dreorth.o dgetu0.o dsafescal.o dblasext.o \
       dlansvd.o printstat.o dgemm_ovwr.o \
       dlansvd_irl.o dbsvd.o dritzvec.o d$(MGS).o second.o

.f.o:
	$(F77) $(FFLAGS) $(IPATH) -c -o $*.o $*.f
.F.o:
	$(F77) $(FFLAGS) $(IPATH) -c -o $*.o $*.F
.c.o:
	$(CC)  $(CFLAGS) $(IPATH) -c -o $*.o $*.c


lib: liblapack libdpropack_$(PLAT).a 


libdpropack_$(PLAT).a : $(DOBJ)
	rm -f libdpropack_$(PLAT).a
	$(MAKELIB) libdpropack_$(PLAT).a $(DOBJ) 
	$(RANLIB)  libdpropack_$(PLAT).a

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
	rm -f  *.o *.il lib*_$(PLAT).a
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

