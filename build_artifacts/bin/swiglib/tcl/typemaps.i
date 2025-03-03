/* -----------------------------------------------------------------------------
 * typemaps.i
 *
 * SWIG typemap library for Tcl8.  This file contains various sorts
 * of typemaps for modifying SWIG's code generation.
 * ----------------------------------------------------------------------------- */

#if !defined(SWIG_USE_OLD_TYPEMAPS)
%include <typemaps/typemaps.swg>
#else

/*
The SWIG typemap library provides a language independent mechanism for
supporting output arguments, input values, and other C function
calling mechanisms.  The primary use of the library is to provide a
better interface to certain C function--especially those involving
pointers.
*/

// INPUT typemaps.
// These remap a C pointer to be an "INPUT" value which is passed by value
// instead of reference.

/*
The following methods can be applied to turn a pointer into a simple
"input" value.  That is, instead of passing a pointer to an object,
you would use a real value instead.

         int            *INPUT
         short          *INPUT
         long           *INPUT
         long long      *INPUT
         unsigned int   *INPUT
         unsigned short *INPUT
         unsigned long  *INPUT
         unsigned long long *INPUT
         unsigned char  *INPUT
         bool           *INPUT
         float          *INPUT
         double         *INPUT
         
To use these, suppose you had a C function like this :

        double fadd(double *a, double *b) {
               return *a+*b;
        }

You could wrap it with SWIG as follows :
        
        %include typemaps.i
        double fadd(double *INPUT, double *INPUT);

or you can use the %apply directive :

        %include typemaps.i
        %apply double *INPUT { double *a, double *b };
        double fadd(double *a, double *b);

*/

%typemap(in) double *INPUT(double temp), double &INPUT(double temp)
{
  if (Tcl_GetDoubleFromObj(interp,$input,&temp) == TCL_ERROR) {
    SWIG_fail;
  }
  $1 = &temp;
}

%typemap(in) float *INPUT(double dvalue, float  temp), float &INPUT(double dvalue, float temp) 
{
  if (Tcl_GetDoubleFromObj(interp,$input,&dvalue) == TCL_ERROR) {
    SWIG_fail;
  }
  temp = (float) dvalue;
  $1 = &temp;
}

%typemap(in) int  *INPUT(int temp), int &INPUT(int temp)
{
  if (Tcl_GetIntFromObj(interp,$input,&temp) == TCL_ERROR) {
    SWIG_fail;
  }
  $1 = &temp;
}

%typemap(in) short *INPUT(int ivalue, short temp), short &INPUT(int ivalue, short temp)
{
  if (Tcl_GetIntFromObj(interp,$input,&ivalue) == TCL_ERROR) {
    SWIG_fail;
  }
  temp = (short) ivalue;
  $1 = &temp;
}

%typemap(in) long *INPUT(int ivalue, long temp), long &INPUT(int ivalue, long temp)
{
  if (Tcl_GetIntFromObj(interp,$input,&ivalue) == TCL_ERROR) {
    SWIG_fail;
  }
  temp = (long) ivalue;
  $1 = &temp;
}

%typemap(in) unsigned int  *INPUT(int ivalue, unsigned int temp), 
             unsigned int  &INPUT(int ivalue, unsigned int temp)
{
  if (Tcl_GetIntFromObj(interp,$input,&ivalue) == TCL_ERROR) {
    SWIG_fail;
  }
  temp = (unsigned int) ivalue;
  $1 = &temp;
}

%typemap(in) unsigned short *INPUT(int ivalue, unsigned short temp),
             unsigned short &INPUT(int ivalue, unsigned short temp)
{
  if (Tcl_GetIntFromObj(interp,$input,&ivalue) == TCL_ERROR) {
    SWIG_fail;
  }
  temp = (unsigned short) ivalue;
  $1 = &temp;
}

%typemap(in) unsigned long *INPUT(int ivalue, unsigned long temp),
             unsigned long &INPUT(int ivalue, unsigned long temp)
{
  if (Tcl_GetIntFromObj(interp,$input,&ivalue) == TCL_ERROR) {
    SWIG_fail;
  }
  temp = (unsigned long) ivalue;
  $1 = &temp;
}

%typemap(in) unsigned char *INPUT(int ivalue, unsigned char temp),
             unsigned char &INPUT(int ivalue, unsigned char temp)
{
  if (Tcl_GetIntFromObj(interp,$input,&ivalue) == TCL_ERROR) {
    SWIG_fail;
  }
  temp = (unsigned char) ivalue;
  $1 = &temp;
}

%typemap(in) signed char *INPUT(int ivalue, signed char temp),
             signed char &INPUT(int ivalue, signed char temp)
{
  if (Tcl_GetIntFromObj(interp,$input,&ivalue) == TCL_ERROR) {
    SWIG_fail;
  }
  temp = (signed char) ivalue;
  $1 = &temp;
}

%typemap(in) bool *INPUT(int ivalue, bool temp),
             bool &INPUT(int ivalue, bool temp)
{
  if (Tcl_GetIntFromObj(interp,$input,&ivalue) == TCL_ERROR) {
    SWIG_fail;
  }
  temp = ivalue ? true : false;
  $1 = &temp;
}

%typemap(in) long long *INPUT($*1_ltype temp), 
             long long &INPUT($*1_ltype temp)
{
  temp = ($*1_ltype) strtoll(Tcl_GetStringFromObj($input,NULL),0,0);
  $1 = &temp;
}

%typemap(in) unsigned long long *INPUT($*1_ltype temp), 
             unsigned long long &INPUT($*1_ltype temp)
{
  temp = ($*1_ltype) strtoull(Tcl_GetStringFromObj($input,NULL),0,0);
  $1 = &temp;
}
  
// OUTPUT typemaps.   These typemaps are used for parameters that
// are output only.   The output value is appended to the result as
// a list element.

/*
The following methods can be applied to turn a pointer into an "output"
value.  When calling a function, no input value would be given for
a parameter, but an output value would be returned.  In the case of
multiple output values, they are returned in the form of a Tcl list.

         int            *OUTPUT
         short          *OUTPUT
         long           *OUTPUT
         long long      *OUTPUT
         unsigned int   *OUTPUT
         unsigned short *OUTPUT
         unsigned long  *OUTPUT
         unsigned long long *OUTPUT
         unsigned char  *OUTPUT
         bool           *OUTPUT
         float          *OUTPUT
         double         *OUTPUT
         
For example, suppose you were trying to wrap the modf() function in the
C math library which splits x into integral and fractional parts (and
returns the integer part in one of its parameters).K:

        double modf(double x, double *ip);

You could wrap it with SWIG as follows :

        %include typemaps.i
        double modf(double x, double *OUTPUT);

or you can use the %apply directive :

        %include typemaps.i
        %apply double *OUTPUT { double *ip };
        double modf(double x, double *ip);

The Tcl output of the function would be a list containing both
output values. 

*/

%typemap(in,numinputs=0)     int            *OUTPUT(int temp),
                     short          *OUTPUT(short temp),
                     long           *OUTPUT(long temp),
                     unsigned int   *OUTPUT(unsigned int temp),
                     unsigned short *OUTPUT(unsigned short temp),
                     unsigned long  *OUTPUT(unsigned long temp),
                     unsigned char  *OUTPUT(unsigned char temp),
	             signed char    *OUTPUT(signed char temp),
                     bool           *OUTPUT(bool temp),
                     float          *OUTPUT(float temp),
                     double         *OUTPUT(double temp),
                     long long      *OUTPUT($*1_ltype temp),
                     unsigned long long *OUTPUT($*1_ltype temp),
	             int            &OUTPUT(int temp),
                     short          &OUTPUT(short temp),
                     long           &OUTPUT(long temp),
                     unsigned int   &OUTPUT(unsigned int temp),
                     unsigned short &OUTPUT(unsigned short temp),
                     unsigned long  &OUTPUT(unsigned long temp),
                     signed char    &OUTPUT(signed char temp),
                     bool           &OUTPUT(bool temp),
                     unsigned char  &OUTPUT(unsigned char temp),
                     float          &OUTPUT(float temp),
                     double         &OUTPUT(double temp),
                     long long      &OUTPUT($*1_ltype temp),
                     unsigned long long &OUTPUT($*1_ltype temp)
"$1 = &temp;";

%typemap(argout)     int     *OUTPUT, int &OUTPUT,
                     short   *OUTPUT, short &OUTPUT,
                     long    *OUTPUT, long &OUTPUT,
                     unsigned int   *OUTPUT, unsigned int &OUTPUT,
                     unsigned short *OUTPUT, unsigned short &OUTPUT,
                     unsigned long  *OUTPUT, unsigned long &OUTPUT,
                     unsigned char  *OUTPUT, unsigned char &OUTPUT,
                     signed char    *OUTPUT, signed char  &OUTPUT,
                     bool           *OUTPUT, bool &OUTPUT
{
  Tcl_Obj *o;
  o = Tcl_NewIntObj((int) *($1));
  Tcl_ListObjAppendElement(interp,Tcl_GetObjResult(interp),o);
}

%typemap(argout) float    *OUTPUT, float &OUTPUT,
                 double   *OUTPUT, double &OUTPUT
{
  Tcl_Obj *o;
  o = Tcl_NewDoubleObj((double) *($1));
  Tcl_ListObjAppendElement(interp,Tcl_GetObjResult(interp),o);
}

%typemap(argout) long long *OUTPUT, long long &OUTPUT
{
  char temp[256];
  Tcl_Obj *o;
  sprintf(temp,"%lld",(long long)*($1));
  o = Tcl_NewStringObj(temp,-1);
  Tcl_ListObjAppendElement(interp,Tcl_GetObjResult(interp),o);
}

%typemap(argout) unsigned long long *OUTPUT, unsigned long long &OUTPUT
{
  char temp[256];
  Tcl_Obj *o;
  sprintf(temp,"%llu",(unsigned long long)*($1));
  o = Tcl_NewStringObj(temp,-1);
  Tcl_ListObjAppendElement(interp,Tcl_GetObjResult(interp),o);
}

// INOUT
// Mappings for an argument that is both an input and output
// parameter

/*
The following methods can be applied to make a function parameter both
an input and output value.  This combines the behavior of both the
"INPUT" and "OUTPUT" methods described earlier.  Output values are
returned in the form of a Tcl list.

         int            *INOUT
         short          *INOUT
         long           *INOUT
         long long      *INOUT
         unsigned int   *INOUT
         unsigned short *INOUT
         unsigned long  *INOUT
         unsigned long long *INOUT
         unsigned char  *INOUT
         bool           *INOUT
         float          *INOUT
         double         *INOUT
         
For example, suppose you were trying to wrap the following function :

        void neg(double *x) {
             *x = -(*x);
        }

You could wrap it with SWIG as follows :

        %include typemaps.i
        void neg(double *INOUT);

or you can use the %apply directive :

        %include typemaps.i
        %apply double *INOUT { double *x };
        void neg(double *x);

Unlike C, this mapping does not directly modify the input value (since
this makes no sense in Tcl).  Rather, the modified input value shows
up as the return value of the function.  Thus, to apply this function
to a Tcl variable you might do this :

       set x [neg $x]

*/


%typemap(in) int *INOUT = int *INPUT;
%typemap(in) short *INOUT = short *INPUT;
%typemap(in) long *INOUT = long *INPUT;
%typemap(in) unsigned int *INOUT = unsigned int *INPUT;
%typemap(in) unsigned short *INOUT = unsigned short *INPUT;
%typemap(in) unsigned long *INOUT = unsigned long *INPUT;
%typemap(in) unsigned char *INOUT = unsigned char *INPUT;
%typemap(in) signed char *INOUT = signed char *INPUT;
%typemap(in) bool *INOUT = bool *INPUT;
%typemap(in) float *INOUT = float *INPUT;
%typemap(in) double *INOUT = double *INPUT;
%typemap(in) long long *INOUT = long long *INPUT;
%typemap(in) unsigned long long *INOUT = unsigned long long *INPUT;

%typemap(in) int &INOUT = int &INPUT;
%typemap(in) short &INOUT = short &INPUT;
%typemap(in) long &INOUT = long &INPUT;
%typemap(in) unsigned int &INOUT = unsigned int &INPUT;
%typemap(in) unsigned short &INOUT = unsigned short &INPUT;
%typemap(in) unsigned long &INOUT = unsigned long &INPUT;
%typemap(in) unsigned char &INOUT = unsigned char &INPUT;
%typemap(in) signed char &INOUT = signed char &INPUT;
%typemap(in) bool &INOUT = bool &INPUT;
%typemap(in) float &INOUT = float &INPUT;
%typemap(in) double &INOUT = double &INPUT;
%typemap(in) long long &INOUT = long long &INPUT;
%typemap(in) unsigned long long &INOUT = unsigned long long &INPUT;

%typemap(argout) int *INOUT = int *OUTPUT;
%typemap(argout) short *INOUT = short *OUTPUT;
%typemap(argout) long *INOUT = long *OUTPUT;
%typemap(argout) unsigned int *INOUT = unsigned int *OUTPUT;
%typemap(argout) unsigned short *INOUT = unsigned short *OUTPUT;
%typemap(argout) unsigned long *INOUT = unsigned long *OUTPUT;
%typemap(argout) unsigned char *INOUT = unsigned char *OUTPUT;
%typemap(argout) signed char *INOUT = signed char *OUTPUT;
%typemap(argout) bool *INOUT = bool *OUTPUT;
%typemap(argout) float *INOUT = float *OUTPUT;
%typemap(argout) double *INOUT = double *OUTPUT;
%typemap(argout) long long *INOUT = long long *OUTPUT;
%typemap(argout) unsigned long long *INOUT = unsigned long long *OUTPUT;

%typemap(argout) int &INOUT = int &OUTPUT;
%typemap(argout) short &INOUT = short &OUTPUT;
%typemap(argout) long &INOUT = long &OUTPUT;
%typemap(argout) unsigned int &INOUT = unsigned int &OUTPUT;
%typemap(argout) unsigned short &INOUT = unsigned short &OUTPUT;
%typemap(argout) unsigned long &INOUT = unsigned long &OUTPUT;
%typemap(argout) unsigned char &INOUT = unsigned char &OUTPUT;
%typemap(argout) signed char &INOUT = signed char &OUTPUT;
%typemap(argout) bool &INOUT = bool &OUTPUT;
%typemap(argout) float &INOUT = float &OUTPUT;
%typemap(argout) double &INOUT = double &OUTPUT;
%typemap(argout) long long &INOUT = long long &OUTPUT;
%typemap(argout) unsigned long long &INOUT = unsigned long long &OUTPUT;


/* Overloading information */

%typemap(typecheck) double *INPUT = double;
%typemap(typecheck) bool *INPUT = bool;
%typemap(typecheck) signed char *INPUT = signed char;
%typemap(typecheck) unsigned char *INPUT = unsigned char;
%typemap(typecheck) unsigned long *INPUT = unsigned long;
%typemap(typecheck) unsigned short *INPUT = unsigned short;
%typemap(typecheck) unsigned int *INPUT = unsigned int;
%typemap(typecheck) long *INPUT = long;
%typemap(typecheck) short *INPUT = short;
%typemap(typecheck) int *INPUT = int;
%typemap(typecheck) float *INPUT = float;
%typemap(typecheck) long long *INPUT = long long;
%typemap(typecheck) unsigned long long *INPUT = unsigned long long;

%typemap(typecheck) double &INPUT = double;
%typemap(typecheck) bool &INPUT = bool;
%typemap(typecheck) signed char &INPUT = signed char;
%typemap(typecheck) unsigned char &INPUT = unsigned char;
%typemap(typecheck) unsigned long &INPUT = unsigned long;
%typemap(typecheck) unsigned short &INPUT = unsigned short;
%typemap(typecheck) unsigned int &INPUT = unsigned int;
%typemap(typecheck) long &INPUT = long;
%typemap(typecheck) short &INPUT = short;
%typemap(typecheck) int &INPUT = int;
%typemap(typecheck) float &INPUT = float;
%typemap(typecheck) long long &INPUT = long long;
%typemap(typecheck) unsigned long long &INPUT = unsigned long long;

%typemap(typecheck) double *INOUT = double;
%typemap(typecheck) bool *INOUT = bool;
%typemap(typecheck) signed char *INOUT = signed char;
%typemap(typecheck) unsigned char *INOUT = unsigned char;
%typemap(typecheck) unsigned long *INOUT = unsigned long;
%typemap(typecheck) unsigned short *INOUT = unsigned short;
%typemap(typecheck) unsigned int *INOUT = unsigned int;
%typemap(typecheck) long *INOUT = long;
%typemap(typecheck) short *INOUT = short;
%typemap(typecheck) int *INOUT = int;
%typemap(typecheck) float *INOUT = float;
%typemap(typecheck) long long *INOUT = long long;
%typemap(typecheck) unsigned long long *INOUT = unsigned long long;

%typemap(typecheck) double &INOUT = double;
%typemap(typecheck) bool &INOUT = bool;
%typemap(typecheck) signed char &INOUT = signed char;
%typemap(typecheck) unsigned char &INOUT = unsigned char;
%typemap(typecheck) unsigned long &INOUT = unsigned long;
%typemap(typecheck) unsigned short &INOUT = unsigned short;
%typemap(typecheck) unsigned int &INOUT = unsigned int;
%typemap(typecheck) long &INOUT = long;
%typemap(typecheck) short &INOUT = short;
%typemap(typecheck) int &INOUT = int;
%typemap(typecheck) float &INOUT = float;
%typemap(typecheck) long long &INOUT = long long;
%typemap(typecheck) unsigned long long &INOUT = unsigned long long;

#endif

// --------------------------------------------------------------------
// Special types
// --------------------------------------------------------------------

%include <tclinterp.i>
%include <tclresult.i>
