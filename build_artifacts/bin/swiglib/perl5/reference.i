/* -----------------------------------------------------------------------------
 * reference.i
 *
 * Accept Perl references as pointers
 * ----------------------------------------------------------------------------- */

/*
The following methods make Perl references work like simple C
pointers.  References can only be used for simple input/output
values, not C arrays however.  It should also be noted that 
REFERENCES are specific to Perl and not supported in other
scripting languages at this time.

         int            *REFERENCE
         short          *REFERENCE
         long           *REFERENCE
         unsigned int   *REFERENCE
         unsigned short *REFERENCE
         unsigned long  *REFERENCE
         unsigned char  *REFERENCE
         float          *REFERENCE
         double         *REFERENCE
         
For example, suppose you were trying to wrap the following function :

        void neg(double *x) {
             *x = -(*x);
        }

You could wrap it with SWIG as follows :

        %include reference.i
        void neg(double *REFERENCE);

or you can use the %apply directive :

        %include reference.i
        %apply double *REFERENCE { double *x };
        void neg(double *x);

Unlike the INOUT mapping described in typemaps.i, this approach directly
modifies the value of a Perl reference.  Thus, you could use it
as follows :

       $x = 3;
       neg(\$x);
       print "$x\n";         # Should print out -3.

*/

%typemap(in) double *REFERENCE (double dvalue), double &REFERENCE(double dvalue)
{
  SV *tempsv;
  if (!SvROK($input)) {
    SWIG_croak("expected a reference");
  }
  tempsv = SvRV($input);
  if ((!SvNOK(tempsv)) && (!SvIOK(tempsv))) {
	printf("Received %d\n", SvTYPE(tempsv));
	SWIG_croak("Expected a double reference.");
  }
  dvalue = SvNV(tempsv);
  $1 = &dvalue;
}

%typemap(in) float *REFERENCE (float dvalue), float &REFERENCE(float dvalue)
{
  SV *tempsv;
  if (!SvROK($input)) {
    SWIG_croak("expected a reference");
  }
  tempsv = SvRV($input);
  if ((!SvNOK(tempsv)) && (!SvIOK(tempsv))) {
    SWIG_croak("expected a double reference");
  }
  dvalue = (float) SvNV(tempsv);
  $1 = &dvalue;
}

%typemap(in) int *REFERENCE (int dvalue), int &REFERENCE (int dvalue)
{
  SV *tempsv;
  if (!SvROK($input)) {
    SWIG_croak("expected a reference");
  }
  tempsv = SvRV($input);
  if (!SvIOK(tempsv)) {
    SWIG_croak("expected an integer reference");
  }
  dvalue = SvIV(tempsv);
  $1 = &dvalue;
}

%typemap(in) short *REFERENCE (short dvalue), short &REFERENCE(short dvalue)
{
  SV *tempsv;
  if (!SvROK($input)) {
    SWIG_croak("expected a reference");
  }
  tempsv = SvRV($input);
  if (!SvIOK(tempsv)) {
    SWIG_croak("expected an integer reference");
  }
  dvalue = (short) SvIV(tempsv);
  $1 = &dvalue;
}
%typemap(in) long *REFERENCE (long dvalue), long &REFERENCE(long dvalue)
{
  SV *tempsv;
  if (!SvROK($input)) {
    SWIG_croak("expected a reference");
  }
  tempsv = SvRV($input);
  if (!SvIOK(tempsv)) {
    SWIG_croak("expected an integer reference");
  }
  dvalue = (long) SvIV(tempsv);
  $1 = &dvalue;
}
%typemap(in) unsigned int *REFERENCE (unsigned int dvalue), unsigned int &REFERENCE(unsigned int dvalue)
{
  SV *tempsv;
  if (!SvROK($input)) {
    SWIG_croak("expected a reference");
  }
  tempsv = SvRV($input);
  if (!SvIOK(tempsv)) {
    SWIG_croak("expected an integer reference");
  }
  dvalue = (unsigned int) SvUV(tempsv);
  $1 = &dvalue;
}
%typemap(in) unsigned short *REFERENCE (unsigned short dvalue), unsigned short &REFERENCE(unsigned short dvalue)
{
  SV *tempsv;
  if (!SvROK($input)) {
    SWIG_croak("expected a reference");
  }
  tempsv = SvRV($input);
  if (!SvIOK(tempsv)) {
    SWIG_croak("expected an integer reference");
  }
  dvalue = (unsigned short) SvUV(tempsv);
  $1 = &dvalue;
}
%typemap(in) unsigned long *REFERENCE (unsigned long dvalue), unsigned long &REFERENCE(unsigned long dvalue)
{
  SV *tempsv;
  if (!SvROK($input)) {
    SWIG_croak("expected a reference");
  }
  tempsv = SvRV($input);
  if (!SvIOK(tempsv)) {
    SWIG_croak("expected an integer reference");
  }
  dvalue = (unsigned long) SvUV(tempsv);
  $1 = &dvalue;
}

%typemap(in) unsigned char *REFERENCE (unsigned char dvalue), unsigned char &REFERENCE(unsigned char dvalue)
{
  SV *tempsv;
  if (!SvROK($input)) {
    SWIG_croak("expected a reference");
  }
  tempsv = SvRV($input);
  if (!SvIOK(tempsv)) {
    SWIG_croak("expected an integer reference");
  }
  dvalue = (unsigned char) SvUV(tempsv);
  $1 = &dvalue;
}

%typemap(in) signed char *REFERENCE (signed char dvalue), signed char &REFERENCE(signed char dvalue)
{
  SV *tempsv;
  if (!SvROK($input)) {
    SWIG_croak("expected a reference");
  }
  tempsv = SvRV($input);
  if (!SvIOK(tempsv)) {
    SWIG_croak("expected an integer reference");
  }
  dvalue = (signed char) SvIV(tempsv);
  $1 = &dvalue;
}

%typemap(in) bool *REFERENCE (bool dvalue), bool &REFERENCE(bool dvalue)
{
  SV *tempsv;
  if (!SvROK($input)) {
    SWIG_croak("expected a reference");
  }
  tempsv = SvRV($input);
  if (!SvIOK(tempsv)) {
    SWIG_croak("expected an integer reference");
  }
  dvalue = SvIV(tempsv) ? true : false;
  $1 = &dvalue;
}

%typemap(typecheck) int *REFERENCE, int &REFERENCE,
                    short *REFERENCE, short &REFERENCE,
                    long *REFERENCE, long  &REFERENCE,
                    signed char *REFERENCE, signed char &REFERENCE,
                    bool *REFERENCE, bool &REFERENCE
{
  $1 = SvROK($input) && SvIOK(SvRV($input));
}
%typemap(typecheck) double *REFERENCE, double &REFERENCE,
                    float *REFERENCE, float &REFERENCE
{
  $1 = SvROK($input);
  if($1) {
    SV *tmpsv = SvRV($input);
    $1 = SvNOK(tmpsv) || SvIOK(tmpsv);
  }
}
%typemap(typecheck) unsigned int   *REFERENCE, unsigned int &REFERENCE,
                    unsigned short *REFERENCE, unsigned short &REFERENCE,
                    unsigned long  *REFERENCE, unsigned long &REFERENCE,
                    unsigned char  *REFERENCE, unsigned char &REFERENCE
{
  $1 = SvROK($input);
  if($1) {
    SV *tmpsv = SvRV($input);
    $1 = SvUOK(tmpsv) || SvIOK(tmpsv);
  }
}

%typemap(argout) double *REFERENCE, double &REFERENCE,
                 float  *REFERENCE, float &REFERENCE
{
  SV *tempsv;
  tempsv = SvRV($arg);
  if (!$1) SWIG_croak("expected a reference");
  sv_setnv(tempsv, (double) *$1);
}

%typemap(argout)       int            *REFERENCE, int &REFERENCE,
                       short          *REFERENCE, short &REFERENCE,
                       long           *REFERENCE, long  &REFERENCE,
                       signed char    *REFERENCE, signed char &REFERENCE,
                       bool           *REFERENCE, bool &REFERENCE
{
  SV *tempsv;
  tempsv = SvRV($input);
  if (!$1) SWIG_croak("expected a reference");
  sv_setiv(tempsv, (IV) *$1);
}

%typemap(argout)       unsigned int   *REFERENCE, unsigned int &REFERENCE,
                       unsigned short *REFERENCE, unsigned short &REFERENCE,
                       unsigned long  *REFERENCE, unsigned long &REFERENCE,
                       unsigned char  *REFERENCE, unsigned char &REFERENCE
{
  SV *tempsv;
  tempsv = SvRV($input);
  if (!$1) SWIG_croak("expected a reference");
  sv_setuv(tempsv, (UV) *$1);
}
