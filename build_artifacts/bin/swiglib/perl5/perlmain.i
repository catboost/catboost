/* -----------------------------------------------------------------------------
 * perlmain.i
 *
 * Code to statically rebuild perl5.
 * ----------------------------------------------------------------------------- */

#ifdef AUTODOC
%subsection "perlmain.i"
%text %{
This module provides support for building a new version of the
Perl executable.  This will be necessary on systems that do
not support shared libraries and may be necessary with C++
extensions.  

This module may only build a stripped down version of the
Perl executable.   Thus, it may be necessary (or desirable)
to hand-edit this file for your particular application.  To
do this, simply copy this file from swig_lib/perl5/perlmain.i
to your working directory and make the appropriate modifications.

This library file works with Perl 5.003.  It may work with earlier
versions, but it hasn't been tested.  As far as I know, this
library is C++ safe.
%}
#endif

%{

static void xs_init _((pTHX));
static PerlInterpreter *my_perl;

int perl_eval(char *string) {
  char *argv[2];
  argv[0] = string;
  argv[1] = (char *) 0;
  return perl_call_argv("eval",0,argv);
}

int
main(int argc, char **argv, char **env)
{
    int exitstatus;

    my_perl = perl_alloc();
    if (!my_perl)
       exit(1);
    perl_construct( my_perl );

    exitstatus = perl_parse( my_perl, xs_init, argc, argv, (char **) NULL );
    if (exitstatus)
	exit( exitstatus );

    /* Initialize all of the module variables */

    exitstatus = perl_run( my_perl );

    perl_destruct( my_perl );
    perl_free( my_perl );

    exit( exitstatus );
}

/* Register any extra external extensions */

/* Do not delete this line--writemain depends on it */
/* EXTERN_C void boot_DynaLoader _((CV* cv)); */

static void
xs_init(pTHX)
{
/*  dXSUB_SYS; */
    char *file = __FILE__;
    {
      /*        newXS("DynaLoader::boot_DynaLoader", boot_DynaLoader, file); */
	newXS(SWIG_name, SWIG_init, file);
#ifdef SWIGMODINIT
	SWIGMODINIT
#endif
    }
}

%}
