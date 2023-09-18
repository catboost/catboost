/* -----------------------------------------------------------------------------
 * typemaps.i
 *
 * Pointer and reference handling typemap library
 *
 * These mappings provide support for input/output arguments and common
 * uses for C/C++ pointers and C++ references.
 * ----------------------------------------------------------------------------- */

/*
INPUT typemaps
--------------

These typemaps remap a C pointer or C++ reference to be an "INPUT" value which is
passed by value instead of reference.

The following typemaps can be applied to turn a pointer or reference into a simple
input value.  That is, instead of passing a pointer or reference to an object,
you would use a real value instead.

        bool               *INPUT, bool               &INPUT
        signed char        *INPUT, signed char        &INPUT
        unsigned char      *INPUT, unsigned char      &INPUT
        short              *INPUT, short              &INPUT
        unsigned short     *INPUT, unsigned short     &INPUT
        int                *INPUT, int                &INPUT
        unsigned int       *INPUT, unsigned int       &INPUT
        long               *INPUT, long               &INPUT
        unsigned long      *INPUT, unsigned long      &INPUT
        long long          *INPUT, long long          &INPUT
        unsigned long long *INPUT, unsigned long long &INPUT
        float              *INPUT, float              &INPUT
        double             *INPUT, double             &INPUT
         
To use these, suppose you had a C function like this :

        double fadd(double *a, double *b) {
               return *a+*b;
        }

You could wrap it with SWIG as follows :
        
        %include <typemaps.i>
        double fadd(double *INPUT, double *INPUT);

or you can use the %apply directive :

        %include <typemaps.i>
        %apply double *INPUT { double *a, double *b };
        double fadd(double *a, double *b);

In Go you could then use it like this:
        answer := modulename.Fadd(10.0, 20.0)

There are no char *INPUT typemaps, however you can apply the signed
char * typemaps instead:
        %include <typemaps.i>
        %apply signed char *INPUT {char *input};
        void f(char *input);
*/

%define INPUT_TYPEMAP(TYPE, GOTYPE)
%typemap(gotype) TYPE *INPUT, TYPE &INPUT "GOTYPE"

 %typemap(in) TYPE *INPUT, TYPE &INPUT
%{ $1 = ($1_ltype)&$input; %}

%typemap(out) TYPE *INPUT, TYPE &INPUT ""

%typemap(goout) TYPE *INPUT, TYPE &INPUT ""

%typemap(freearg) TYPE *INPUT, TYPE &INPUT ""

%typemap(argout) TYPE *INPUT, TYPE &INPUT ""

// %typemap(typecheck) TYPE *INPUT = TYPE;
// %typemap(typecheck) TYPE &INPUT = TYPE;
%enddef

INPUT_TYPEMAP(bool, bool);
INPUT_TYPEMAP(signed char, int8);
INPUT_TYPEMAP(char, byte);
INPUT_TYPEMAP(unsigned char, byte);
INPUT_TYPEMAP(short, int16);
INPUT_TYPEMAP(unsigned short, uint16);
INPUT_TYPEMAP(int, int);
INPUT_TYPEMAP(unsigned int, uint);
INPUT_TYPEMAP(long, int64);
INPUT_TYPEMAP(unsigned long, uint64);
INPUT_TYPEMAP(long long, int64);
INPUT_TYPEMAP(unsigned long long, uint64);
INPUT_TYPEMAP(float, float32);
INPUT_TYPEMAP(double, float64);

#undef INPUT_TYPEMAP

// OUTPUT typemaps.   These typemaps are used for parameters that
// are output only.   An array replaces the c pointer or reference parameter. 
// The output value is returned in this array passed in. 

/*
OUTPUT typemaps
---------------

The following typemaps can be applied to turn a pointer or reference
into an "output" value.  When calling a function, no input value would
be given for a parameter, but an output value would be returned.  This
works by a Go slice being passed as a parameter where a c pointer or
reference is required.  As with any Go function, the array is passed
by reference so that any modifications to the array will be picked up
in the calling function.  Note that the array passed in MUST have at
least one element, but as the c function does not require any input,
the value can be set to anything.

        bool               *OUTPUT, bool               &OUTPUT
        signed char        *OUTPUT, signed char        &OUTPUT
        unsigned char      *OUTPUT, unsigned char      &OUTPUT
        short              *OUTPUT, short              &OUTPUT
        unsigned short     *OUTPUT, unsigned short     &OUTPUT
        int                *OUTPUT, int                &OUTPUT
        unsigned int       *OUTPUT, unsigned int       &OUTPUT
        long               *OUTPUT, long               &OUTPUT
        unsigned long      *OUTPUT, unsigned long      &OUTPUT
        long long          *OUTPUT, long long          &OUTPUT
        unsigned long long *OUTPUT, unsigned long long &OUTPUT
        float              *OUTPUT, float              &OUTPUT
        double             *OUTPUT, double             &OUTPUT
         
For example, suppose you were trying to wrap the modf() function in the
C math library which splits x into integral and fractional parts (and
returns the integer part in one of its parameters):

        double modf(double x, double *ip);

You could wrap it with SWIG as follows :

        %include <typemaps.i>
        double modf(double x, double *OUTPUT);

or you can use the %apply directive :

        %include <typemaps.i>
        %apply double *OUTPUT { double *ip };
        double modf(double x, double *ip);

The Go output of the function would be the function return value and the 
value in the single element array. In Go you would use it like this:

    ptr := []float64{0.0}
    fraction := modulename.Modf(5.0,ptr)

There are no char *OUTPUT typemaps, however you can apply the signed
char * typemaps instead:
        %include <typemaps.i>
        %apply signed char *OUTPUT {char *output};
        void f(char *output);
*/

%define OUTPUT_TYPEMAP(TYPE, GOTYPE)
%typemap(gotype) TYPE *OUTPUT, TYPE &OUTPUT %{[]GOTYPE%}

%typemap(in) TYPE *OUTPUT($*1_ltype temp), TYPE &OUTPUT($*1_ltype temp)
{
  if ($input.len == 0) {
    _swig_gopanic("array must contain at least 1 element");
  }
  $1 = &temp;
}

%typemap(out) TYPE *OUTPUT, TYPE &OUTPUT ""

%typemap(goout) TYPE *INPUT, TYPE &INPUT ""

%typemap(freearg) TYPE *OUTPUT, TYPE &OUTPUT ""

%typemap(argout) TYPE *OUTPUT, TYPE &OUTPUT
{
  TYPE* a = (TYPE *) $input.array;
  a[0] = temp$argnum;
}

%enddef

OUTPUT_TYPEMAP(bool, bool);
OUTPUT_TYPEMAP(signed char, int8);
OUTPUT_TYPEMAP(char, byte);
OUTPUT_TYPEMAP(unsigned char, byte);
OUTPUT_TYPEMAP(short, int16);
OUTPUT_TYPEMAP(unsigned short, uint16);
OUTPUT_TYPEMAP(int, int);
OUTPUT_TYPEMAP(unsigned int, uint);
OUTPUT_TYPEMAP(long, int64);
OUTPUT_TYPEMAP(unsigned long, uint64);
OUTPUT_TYPEMAP(long long, int64);
OUTPUT_TYPEMAP(unsigned long long, uint64);
OUTPUT_TYPEMAP(float, float32);
OUTPUT_TYPEMAP(double, float64);

#undef OUTPUT_TYPEMAP

/*
INOUT typemaps
--------------

Mappings for a parameter that is both an input and an output parameter

The following typemaps can be applied to make a function parameter both
an input and output value.  This combines the behavior of both the
"INPUT" and "OUTPUT" typemaps described earlier.  Output values are
returned as an element in a Go slice.

        bool               *INOUT, bool               &INOUT
        signed char        *INOUT, signed char        &INOUT
        unsigned char      *INOUT, unsigned char      &INOUT
        short              *INOUT, short              &INOUT
        unsigned short     *INOUT, unsigned short     &INOUT
        int                *INOUT, int                &INOUT
        unsigned int       *INOUT, unsigned int       &INOUT
        long               *INOUT, long               &INOUT
        unsigned long      *INOUT, unsigned long      &INOUT
        long long          *INOUT, long long          &INOUT
        unsigned long long *INOUT, unsigned long long &INOUT
        float              *INOUT, float              &INOUT
        double             *INOUT, double             &INOUT
         
For example, suppose you were trying to wrap the following function :

        void neg(double *x) {
             *x = -(*x);
        }

You could wrap it with SWIG as follows :

        %include <typemaps.i>
        void neg(double *INOUT);

or you can use the %apply directive :

        %include <typemaps.i>
        %apply double *INOUT { double *x };
        void neg(double *x);

This works similarly to C in that the mapping directly modifies the
input value - the input must be an array with a minimum of one element. 
The element in the array is the input and the output is the element in 
the array.

       x := []float64{5.0}
       Neg(x);

The implementation of the OUTPUT and INOUT typemaps is different to
other languages in that other languages will return the output value
as part of the function return value. This difference is due to Go
being a typed language.

There are no char *INOUT typemaps, however you can apply the signed
char * typemaps instead:
        %include <typemaps.i>
        %apply signed char *INOUT {char *inout};
        void f(char *inout);
*/

%define INOUT_TYPEMAP(TYPE, GOTYPE)
%typemap(gotype) TYPE *INOUT, TYPE &INOUT %{[]GOTYPE%}

%typemap(in) TYPE *INOUT, TYPE &INOUT {
  if ($input.len == 0) {
    _swig_gopanic("array must contain at least 1 element");
  }
  $1 = ($1_ltype) $input.array;
}

%typemap(out) TYPE *INOUT, TYPE &INOUT ""

%typemap(goout) TYPE *INOUT, TYPE &INOUT ""

%typemap(freearg) TYPE *INOUT, TYPE &INOUT ""

%typemap(argout) TYPE *INOUT, TYPE &INOUT ""

%enddef

INOUT_TYPEMAP(bool, bool);
INOUT_TYPEMAP(signed char, int8);
INOUT_TYPEMAP(char, byte);
INOUT_TYPEMAP(unsigned char, byte);
INOUT_TYPEMAP(short, int16);
INOUT_TYPEMAP(unsigned short, uint16);
INOUT_TYPEMAP(int, int);
INOUT_TYPEMAP(unsigned int, uint);
INOUT_TYPEMAP(long, int64);
INOUT_TYPEMAP(unsigned long, uint64);
INOUT_TYPEMAP(long long, int64);
INOUT_TYPEMAP(unsigned long long, uint64);
INOUT_TYPEMAP(float, float32);
INOUT_TYPEMAP(double, float64);

#undef INOUT_TYPEMAP
