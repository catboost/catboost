#
#     (C) Rasmus Munk Larsen, Stanford University, 2004
#
include ../make.inc
IPATH = -I.

all: lib exampleprogs

### Object files.

DOBJ = clanbpro.o creorth.o cgetu0.o csafescal.o sblasext.o cblasext.o \
      clansvd.o printstat.o cgemm_ovwr.o sgemm_ovwr.o  \
      clansvd_irl.o sbsvd.o critzvec.o c$(MGS).o  second.o 

.f.o:
	$(F77) $(FFLAGS) $(IPATH) -c -o $*.o $*.f
.F.o:
	$(F77) $(FFLAGS) $(IPATH) -c -o $*.o $*.F
.c.o:
	$(CC)  $(CFLAGS) $(IPATH) -c -o $*.o $*.c


lib: liblapack libcpropack_$(PLAT).a 

libcpropack_$(PLAT).a : $(DOBJ)
	rm -f libcpropack_$(PLAT).a
	$(MAKELIB) libcpropack_$(PLAT).a $(DOBJ) 
	$(RANLIB)  libcpropack_$(PLAT).a

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

