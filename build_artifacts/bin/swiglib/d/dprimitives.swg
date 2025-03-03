/* -----------------------------------------------------------------------------
 * dprimitves.swg
 *
 * Typemaps for primitive types.
 * ----------------------------------------------------------------------------- */

// C long/ulong width depends on the target arch, use stdlib aliases for them.
#if (SWIG_D_VERSION == 1)
%pragma(d) imdmoduleimports = "static import tango.stdc.config;"
%pragma(d) globalproxyimports = "static import tango.stdc.config;"
#define SWIG_LONG_DTYPE tango.stdc.config.c_long
#define SWIG_ULONG_DTYPE tango.stdc.config.c_ulong
#else
%pragma(d) imdmoduleimports = "static import core.stdc.config;"
%pragma(d) globalproxyimports = "static import core.stdc.config;"
#define SWIG_LONG_DTYPE core.stdc.config.c_long
#define SWIG_ULONG_DTYPE core.stdc.config.c_ulong
#endif

/*
 * The SWIG_D_PRIMITIVE macro is used to define the typemaps for the primitive
 * types, because are more or less the same for all of them. The few special
 * cases are handled below.
 */
%define SWIG_D_PRIMITIVE(TYPE, DTYPE)
%typemap(ctype) TYPE, const TYPE & "TYPE"
%typemap(imtype) TYPE, const TYPE & "DTYPE"
%typemap(dtype, cprimitive="1") TYPE, const TYPE & "DTYPE"

%typemap(in) TYPE "$1 = ($1_ltype)$input;"
%typemap(out) TYPE "$result = $1;"
%typemap(directorin) TYPE "$input = $1;"
%typemap(directorout) TYPE "$result = ($1_ltype)$input;"
%typemap(ddirectorin) TYPE "$winput"
%typemap(ddirectorout) TYPE "$dcall"

%typemap(in) const TYPE & ($*1_ltype temp)
%{ temp = ($*1_ltype)$input;
   $1 = &temp; %}
%typemap(out) const TYPE & "$result = *$1;"
%typemap(directorin) const TYPE & "$input = $1;"
%typemap(directorout,warning=SWIGWARN_TYPEMAP_THREAD_UNSAFE_MSG) const TYPE &
%{ static $*1_ltype temp;
   temp = ($*1_ltype)$input;
   $result = &temp; %}
%typemap(ddirectorin) const TYPE & "$winput"
%typemap(ddirectorout) const TYPE & "$dcall"

%typemap(din) TYPE, const TYPE & "$dinput"
%typemap(dout, excode=SWIGEXCODE) TYPE, const TYPE & {
  auto ret = $imcall;$excode
  return ret;
}
%enddef


SWIG_D_PRIMITIVE(bool, bool)
SWIG_D_PRIMITIVE(char, char)
SWIG_D_PRIMITIVE(signed char, byte)
SWIG_D_PRIMITIVE(unsigned char, ubyte)
SWIG_D_PRIMITIVE(short, short)
SWIG_D_PRIMITIVE(unsigned short, ushort)
SWIG_D_PRIMITIVE(int, int)
SWIG_D_PRIMITIVE(unsigned int, uint)
SWIG_D_PRIMITIVE(long, SWIG_LONG_DTYPE)
SWIG_D_PRIMITIVE(unsigned long, SWIG_ULONG_DTYPE)
SWIG_D_PRIMITIVE(size_t, size_t)
SWIG_D_PRIMITIVE(long long, long)
SWIG_D_PRIMITIVE(unsigned long long, ulong)
SWIG_D_PRIMITIVE(float, float)
SWIG_D_PRIMITIVE(double, double)


// The C++ boolean type needs some special casing since it is not part of the
// C standard and is thus represented as unsigned int in the C wrapper layer.
%typemap(ctype) bool, const bool & "unsigned int"
%typemap(imtype) bool, const bool & "uint"

%typemap(in) bool "$1 = $input ? true : false;"
%typemap(in) const bool & ($*1_ltype temp)
%{ temp = $input ? true : false;
   $1 = &temp; %}

%typemap(directorout) bool
  "$result = $input ? true : false;"
%typemap(directorout,warning=SWIGWARN_TYPEMAP_THREAD_UNSAFE_MSG) const bool &
%{ static $*1_ltype temp;
   temp = $input ? true : false;
   $result = &temp; %}

%typemap(ddirectorin) bool "($winput ? true : false)"

%typemap(dout, excode=SWIGEXCODE) bool, const bool & {
  bool ret = $imcall ? true : false;$excode
  return ret;
}


// Judging from the history of the C# module, the explicit casts are needed for
// certain versions of VC++.
%typemap(out) unsigned long "$result = (unsigned long)$1;"
%typemap(out) const unsigned long & "$result = (unsigned long)*$1;"


/*
 * Typecheck typemaps.
 */

%typecheck(SWIG_TYPECHECK_BOOL)
    bool,
    const bool &
    ""

%typecheck(SWIG_TYPECHECK_CHAR)
    char,
    const char &
    ""

%typecheck(SWIG_TYPECHECK_INT8)
    signed char,
    const signed char &
    ""

%typecheck(SWIG_TYPECHECK_UINT8)
    unsigned char,
    const unsigned char &
    ""

%typecheck(SWIG_TYPECHECK_INT16)
    short,
    const short &
    ""

%typecheck(SWIG_TYPECHECK_UINT16)
    unsigned short,
    const unsigned short &
    ""

%typecheck(SWIG_TYPECHECK_INT32)
    int,
    long,
    const int &,
    const long &
    ""

%typecheck(SWIG_TYPECHECK_UINT32)
    unsigned int,
    unsigned long,
    const unsigned int &,
    const unsigned long &
    ""

%typecheck(SWIG_TYPECHECK_INT64)
    long long,
    const long long &
    ""

%typecheck(SWIG_TYPECHECK_UINT64)
    unsigned long long,
    const unsigned long long &
    ""

%typecheck(SWIG_TYPECHECK_FLOAT)
    float,
    const float &
    ""

%typecheck(SWIG_TYPECHECK_DOUBLE)
    double,
    const double &
    ""
