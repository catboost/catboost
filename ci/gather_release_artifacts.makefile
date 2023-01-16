REV = 0.16.4
UDIR = upload

upload += catboost-darwin-${REV}
upload += catboost-linux-${REV}
upload += catboost-${REV}.exe
upload += catboost-R-Darwin-${REV}.tgz
upload += catboost-R-Linux-${REV}.tgz
upload += catboost-R-Windows-${REV}.tgz
upload += libcatboostr.dll
upload += libcatboostr-linux.so
upload += libcatboostr-darwin.so
upload += catboostmodel.dll
upload += libcatboostmodel.so
upload += libcatboostmodel.dylib
upload += catboostmodel.lib

all: $(addprefix ${UDIR}/, ${upload})

${UDIR}/catboost-darwin-${REV}: mac_bin/bin/catboost | ${UDIR}/.
	cp -p $< $@
	md5sum $@
	a=`md5sum $@` ; grep $$a *_BuildAll_*_md5.checksum

${UDIR}/catboost-linux-${REV}: linux_bin/bin/catboost | ${UDIR}/.
	cp -p $< $@
	md5sum $@
	a=`md5sum $@` ; grep $$a *_BuildAll_*_md5.checksum

${UDIR}/catboost-${REV}.exe: bin/catboost.exe | ${UDIR}/.
	cp -p $< $@
	md5sum $@
	a=`md5sum $@` ; grep $$a *_BuildAll_*_md5.checksum

${UDIR}/catboost-R-%-${REV}.tgz: R/catboost-R-%.tgz
	cp -p $< $@
	md5sum $@
	a=`md5sum $@` ; grep $$a *_BuildAll_*_md5.checksum

${UDIR}/libcatboostr.dll: R/catboost-R-Windows.tgz
	echo catboost/inst/libs/x64/${@F} >T
	tar xvzp -C ${UDIR} --strip-components 4 -T T -f $<
	touch $@

${UDIR}/libcatboostr-linux.so: R/catboost-R-Linux.tgz
	echo catboost/inst/libs/$(subst -linux,,${@F}) >T
	cd ${UDIR} && tar xvzp --strip-components 3 -T ../T -f ../$<
	cd ${UDIR} && mv $(subst -linux,,${@F}) ${@F}
	touch $@

${UDIR}/libcatboostr-darwin.so: R/catboost-R-Darwin.tgz
	echo catboost/inst/libs/$(subst -darwin,,${@F}) >T
	cd ${UDIR} && tar xvzp --strip-components 3 -T ../T -f ../$<
	cd ${UDIR} && mv $(subst -darwin,,${@F}) ${@F}
	touch $@

${UDIR}/catboostmodel.dll: model_interface_win/model_interface/catboostmodel.dll
	cp -p $< $@
	md5sum $@
	a=`md5sum $@` ; grep $$a *_BuildAll_*_md5.checksum

${UDIR}/libcatboostmodel.so: ./model_interface_lin/model_interface/libcatboostmodel.so
	cp -p $< $@
	md5sum $@
	a=`md5sum $@` ; grep $$a *_BuildAll_*_md5.checksum

${UDIR}/libcatboostmodel.dylib: ./model_interface_mac/model_interface/libcatboostmodel.dylib
	cp -p $< $@
	md5sum $@
	a=`md5sum $@` ; grep $$a *_BuildAll_*_md5.checksum

${UDIR}/catboostmodel.lib: model_interface_win/model_interface/catboostmodel.lib
	cp -p $< $@
	md5sum $@
	a=`md5sum $@` ; grep $$a *_BuildAll_*_md5.checksum

%/.: ; mkdir -p $@
