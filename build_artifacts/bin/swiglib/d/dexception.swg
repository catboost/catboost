/* -----------------------------------------------------------------------------
 * dexception.swg
 *
 * Typemaps used for propagating C++ exceptions to D.
 * ----------------------------------------------------------------------------- */

// Code which is inserted into the dout typemaps and class constructors via
// excode if exceptions can be thrown.
%define SWIGEXCODE "\n  if ($imdmodule.SwigPendingException.isPending) throw $imdmodule.SwigPendingException.retrieve();" %enddef

%typemap(throws, canthrow=1) int,
                 long,
                 short,
                 unsigned int,
                 unsigned long,
                 unsigned short
%{ char error_msg[256];
   sprintf(error_msg, "C++ $1_type exception thrown, value: %d", $1);
   SWIG_DSetPendingException(SWIG_DException, error_msg);
   return $null; %}

%typemap(throws, canthrow=1) SWIGTYPE, SWIGTYPE &, SWIGTYPE *, SWIGTYPE [ANY],
  enum SWIGTYPE, const enum SWIGTYPE &
%{ (void)$1;
   SWIG_DSetPendingException(SWIG_DException, "C++ $1_type exception thrown");
   return $null; %}

%typemap(throws, canthrow=1) char *
%{ SWIG_DSetPendingException(SWIG_DException, $1);
   return $null; %}
