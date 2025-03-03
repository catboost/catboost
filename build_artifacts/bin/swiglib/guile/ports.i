/* -----------------------------------------------------------------------------
 * ports.i
 *
 * Guile typemaps for handling ports
 * ----------------------------------------------------------------------------- */

%{
  #ifndef _POSIX_SOURCE
  /* This is needed on Solaris for fdopen(). */
  #  define _POSIX_SOURCE 199506L
  #endif
  #include <stdio.h>
  #include <errno.h>
  #include <unistd.h>
%}

/* This typemap for FILE * accepts
   (1) FILE * pointer objects,
   (2) Scheme file ports.  In this case, it creates a temporary C stream
       which reads or writes from a dup'ed file descriptor.
 */

%typemap(in, doc="$NAME is a file port or a FILE * pointer") FILE *
{
  if (SWIG_ConvertPtr($input, (void**) &($1), $1_descriptor, 0) != 0) {
    if (!(SCM_FPORTP($input))) {
      scm_wrong_type_arg("$symname", $argnum, $input);
    } else {
      int fd;
      if (SCM_OUTPUT_PORT_P($input)) {
        scm_force_output($input);
      }
      fd=dup(SCM_FPORT_FDES($input));
      if (fd==-1) {
        scm_misc_error("$symname", strerror(errno), SCM_EOL);
      }
      $1=fdopen(fd, SCM_OUTPUT_PORT_P($input) ? (SCM_INPUT_PORT_P($input) ? "r+" : "w") : "r");
      if ($1==NULL) {
        scm_misc_error("$symname", strerror(errno), SCM_EOL);
      }
    }
  }
}

%typemap(freearg) FILE*  {
  if ($1) {
    fclose($1);
  }
}

