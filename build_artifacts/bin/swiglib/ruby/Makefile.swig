# File : Makefile.swig
# Makefile for a SWIG module.  Use this file if you are
# producing a Ruby extension for general use or distribution.
#
# 1.  Prepare extconf.rb.
# 2.  Modify this file as appropriate.
# 3.  Type 'make -f Makefile.swig' to generate wrapper code and Makefile.
# 4.  Type 'make' to build your extension.
# 5.  Type 'make install' to install your extension.
#

MODULE    = yourmodule
FEATURE   = $(MODULE)
INTERFACE = $(MODULE).i
RUBY      = ruby
SWIG      = swig

# for C extension
SWIGOPT   = -ruby
WRAPPER   = $(MODULE)_wrap.c

## for C++ extension
#SWIGOPT   = -ruby -c++
#WRAPPER   = $(MODULE)_wrap.cc


swigall: $(WRAPPER) Makefile

$(WRAPPER): $(INTERFACE)
	$(SWIG) $(SWIGOPT) -o $@ $(INTERFACE)

Makefile: extconf.rb
	$(RUBY) extconf.rb
	@if [ -f Makefile ] ; then\
		echo "include Makefile.swig" >> Makefile;\
	fi

swigclean:
	@if [ -f Makefile ] ; then\
		make -f Makefile clean;\
	fi
	rm -f Makefile $(WRAPPER)
