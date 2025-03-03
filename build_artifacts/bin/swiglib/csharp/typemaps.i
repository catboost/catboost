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

These typemaps are used for pointer/reference parameters that are input only
and are mapped to a C# input parameter.

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

In C# you could then use it like this:
        double answer = modulename.fadd(10.0, 20.0);
*/

%define INPUT_TYPEMAP(TYPE, CTYPE, CSTYPE)
%typemap(ctype, out="void *") TYPE *INPUT, TYPE &INPUT "CTYPE"
%typemap(imtype, out="global::System.IntPtr") TYPE *INPUT, TYPE &INPUT "CSTYPE"
%typemap(cstype, out="$csclassname") TYPE *INPUT, TYPE &INPUT "CSTYPE"
%typemap(csin) TYPE *INPUT, TYPE &INPUT "$csinput"

%typemap(in) TYPE *INPUT, TYPE &INPUT
%{ $1 = ($1_ltype)&$input; %}

%typemap(typecheck) TYPE *INPUT = TYPE;
%typemap(typecheck) TYPE &INPUT = TYPE;
%enddef

INPUT_TYPEMAP(bool,               unsigned int,         bool)
//INPUT_TYPEMAP(char,               char,                 char)
INPUT_TYPEMAP(signed char,        signed char,          sbyte)
INPUT_TYPEMAP(unsigned char,      unsigned char,        byte)
INPUT_TYPEMAP(short,              short,                short)
INPUT_TYPEMAP(unsigned short,     unsigned short,       ushort)
INPUT_TYPEMAP(int,                int,                  int)
INPUT_TYPEMAP(unsigned int,       unsigned int,         uint)
INPUT_TYPEMAP(long,               long,                 int)
INPUT_TYPEMAP(unsigned long,      unsigned long,        uint)
INPUT_TYPEMAP(long long,          long long,            long)
INPUT_TYPEMAP(unsigned long long, unsigned long long,   ulong)
INPUT_TYPEMAP(float,              float,                float)
INPUT_TYPEMAP(double,             double,               double)

#undef INPUT_TYPEMAP

/*
OUTPUT typemaps
---------------

These typemaps are used for pointer/reference parameters that are output only and
are mapped to a C# output parameter.

The following typemaps can be applied to turn a pointer or reference into an "output"
value.  When calling a function, no input value would be given for
a parameter, but an output value would be returned. In C#, the 'out' keyword is
used when passing the parameter to a function that takes an output parameter.

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

The C# output of the function would be the function return value and the 
value returned in the second output parameter. In C# you would use it like this:

    double dptr;
    double fraction = modulename.modf(5, out dptr);
*/

%define OUTPUT_TYPEMAP(TYPE, CTYPE, CSTYPE, TYPECHECKPRECEDENCE)
%typemap(ctype, out="void *") TYPE *OUTPUT, TYPE &OUTPUT "CTYPE *"
%typemap(imtype, out="global::System.IntPtr") TYPE *OUTPUT, TYPE &OUTPUT "out CSTYPE"
%typemap(cstype, out="$csclassname") TYPE *OUTPUT, TYPE &OUTPUT "out CSTYPE"
%typemap(csin) TYPE *OUTPUT, TYPE &OUTPUT "out $csinput"

%typemap(in) TYPE *OUTPUT, TYPE &OUTPUT
%{ $1 = ($1_ltype)$input; %}

%typecheck(SWIG_TYPECHECK_##TYPECHECKPRECEDENCE) TYPE *OUTPUT, TYPE &OUTPUT ""
%enddef

OUTPUT_TYPEMAP(bool,               unsigned int,         bool,     BOOL_PTR)
//OUTPUT_TYPEMAP(char,               char,                 char,     CHAR_PTR)
OUTPUT_TYPEMAP(signed char,        signed char,          sbyte,    INT8_PTR)
OUTPUT_TYPEMAP(unsigned char,      unsigned char,        byte,     UINT8_PTR)
OUTPUT_TYPEMAP(short,              short,                short,    INT16_PTR)
OUTPUT_TYPEMAP(unsigned short,     unsigned short,       ushort,   UINT16_PTR)
OUTPUT_TYPEMAP(int,                int,                  int,      INT32_PTR)
OUTPUT_TYPEMAP(unsigned int,       unsigned int,         uint,     UINT32_PTR)
OUTPUT_TYPEMAP(long,               long,                 int,      INT32_PTR)
OUTPUT_TYPEMAP(unsigned long,      unsigned long,        uint,     UINT32_PTR)
OUTPUT_TYPEMAP(long long,          long long,            long,     INT64_PTR)
OUTPUT_TYPEMAP(unsigned long long, unsigned long long,   ulong,    UINT64_PTR)
OUTPUT_TYPEMAP(float,              float,                float,    FLOAT_PTR)
OUTPUT_TYPEMAP(double,             double,               double,   DOUBLE_PTR)

#undef OUTPUT_TYPEMAP

%typemap(in) bool *OUTPUT, bool &OUTPUT
%{ *$input = 0; 
   $1 = ($1_ltype)$input; %}


/*
INOUT typemaps
--------------

These typemaps are for pointer/reference parameters that are both input and
output and are mapped to a C# reference parameter.

The following typemaps can be applied to turn a pointer or reference into a
reference parameters, that is the parameter is both an input and an output.
In C#, the 'ref' keyword is used for reference parameters.

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

The C# output of the function would be the new value returned by the 
reference parameter. In C# you would use it like this:


       double x = 5.0;
       neg(ref x);

The implementation of the OUTPUT and INOUT typemaps is different to the scripting
languages in that the scripting languages will return the output value as part 
of the function return value.

*/

%define INOUT_TYPEMAP(TYPE, CTYPE, CSTYPE, TYPECHECKPRECEDENCE)
%typemap(ctype, out="void *") TYPE *INOUT, TYPE &INOUT "CTYPE *"
%typemap(imtype, out="global::System.IntPtr") TYPE *INOUT, TYPE &INOUT "ref CSTYPE"
%typemap(cstype, out="$csclassname") TYPE *INOUT, TYPE &INOUT "ref CSTYPE"
%typemap(csin) TYPE *INOUT, TYPE &INOUT "ref $csinput"

%typemap(in) TYPE *INOUT, TYPE &INOUT
%{ $1 = ($1_ltype)$input; %}

%typecheck(SWIG_TYPECHECK_##TYPECHECKPRECEDENCE) TYPE *INOUT, TYPE &INOUT ""
%enddef

INOUT_TYPEMAP(bool,               unsigned int,         bool,     BOOL_PTR)
//INOUT_TYPEMAP(char,               char,                 char,     CHAR_PTR)
INOUT_TYPEMAP(signed char,        signed char,          sbyte,    INT8_PTR)
INOUT_TYPEMAP(unsigned char,      unsigned char,        byte,     UINT8_PTR)
INOUT_TYPEMAP(short,              short,                short,    INT16_PTR)
INOUT_TYPEMAP(unsigned short,     unsigned short,       ushort,   UINT16_PTR)
INOUT_TYPEMAP(int,                int,                  int,      INT32_PTR)
INOUT_TYPEMAP(unsigned int,       unsigned int,         uint,     UINT32_PTR)
INOUT_TYPEMAP(long,               long,                 int,      INT32_PTR)
INOUT_TYPEMAP(unsigned long,      unsigned long,        uint,     UINT32_PTR)
INOUT_TYPEMAP(long long,          long long,            long,     INT64_PTR)
INOUT_TYPEMAP(unsigned long long, unsigned long long,   ulong,    UINT64_PTR)
INOUT_TYPEMAP(float,              float,                float,    FLOAT_PTR)
INOUT_TYPEMAP(double,             double,               double,   DOUBLE_PTR)

#undef INOUT_TYPEMAP

