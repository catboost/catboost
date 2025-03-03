# File : Makefile.pl
# MakeMaker file for a SWIG module.  Use this file if you are
# producing a module for general use or distribution.
#
# 1.  Modify the file as appropriate. Replace $module with the
#     real name of your module and wrapper file.
# 2.  Run perl as 'perl Makefile.pl'
# 3.  Type 'make' to build your module
# 4.  Type 'make install' to install your module.
#
# See "Programming Perl", 2nd. Ed, for more gory details than
# you ever wanted to know.

use ExtUtils::MakeMaker;
WriteMakefile(
     'NAME' => '$module',            # Name of your module
     'LIBS' => [''],                 # Custom libraries (if any)
     'OBJECT' => '$module_wrap.o'    # Object files
);
