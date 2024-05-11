#
#     (C) Rasmus Munk Larsen, Stanford University, 2004
#
include ./make.inc
IPATH = -I.

all: 
	@( cd double; \
	$(MAKE) all; \
	cd .. ; \
	cd single; \
	$(MAKE) all; \
	cd .. ; \
	cd complex16; \
	$(MAKE) all; \
	cd .. ; \
	cd complex8; \
	$(MAKE) all; \
	cd .. )		

test: 
	@( cd double; \
	$(MAKE) test; \
	cd .. ; \
	cd single; \
	$(MAKE) test; \
	cd .. ; \
	cd complex16; \
	$(MAKE) test; \
	cd .. ; \
	cd complex8; \
	$(MAKE) test; \
	cd .. )		

verify: 
	@( cd double; \
	$(MAKE) verify; \
	cd .. ; \
	cd single; \
	$(MAKE) verify; \
	cd .. ; \
	cd complex16; \
	$(MAKE) verify; \
	cd .. ; \
	cd complex8; \
	$(MAKE) verify; \
	cd .. )		


clean:	
	rm -f  *.o *.il
	rm -rf rii_files
	@( cd double; \
	$(MAKE) clean; \
	cd .. ; \
	cd single; \
	$(MAKE) clean; \
	cd .. ; \
	cd complex16; \
	$(MAKE) clean; \
	cd .. ; \
	cd complex8; \
	$(MAKE) clean; \
	cd .. )

cleanall:	
	rm -f  *.o *.il *~
	rm -rf rii_files
	@( cd Make; \
	rm -f *~ ; \
	cd .. ; \
	cd double; \
	$(MAKE) cleanall; \
	cd .. ; \
	cd single; \
	$(MAKE) cleanall; \
	cd .. ; \
	cd complex16; \
	$(MAKE) cleanall; \
	cd .. ; \
	cd complex8; \
	$(MAKE) cleanall; \
	cd .. )

