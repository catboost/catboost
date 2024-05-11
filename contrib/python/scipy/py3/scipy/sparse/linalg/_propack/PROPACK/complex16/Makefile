#
#     (C) Rasmus Munk Larsen, Stanford University, 2004
#
include ../make.inc
IPATH = -I.

all: lib exampleprogs

### Object files.

DOBJ = zlanbpro.o zreorth.o zgetu0.o zsafescal.o dblasext.o zblasext.o \
      zlansvd.o printstat.o zgemm_ovwr.o dgemm_ovwr.o  \
      zlansvd_irl.o dbsvd.o zritzvec.o z$(MGS).o  second.o 

.f.o:
	$(F77) $(FFLAGS) $(IPATH) -c -o $*.o $*.f
.F.o:
	$(F77) $(FFLAGS) $(IPATH) -c -o $*.o $*.F
.c.o:
	$(CC)  $(CFLAGS) $(IPATH) -c -o $*.o $*.c


lib:  liblapack libzpropack_$(PLAT).a 

libzpropack_$(PLAT).a: $(DOBJ)
	rm -f libzpropack_$(PLAT).a
	$(MAKELIB) libzpropack_$(PLAT).a $(DOBJ) 
	$(RANLIB)  libzpropack_$(PLAT).a

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

