/* --------------------------------------------------------------------------
 *
 *  Arrays typemaps
 *
 * --------------------------------------------------------------------------*/

%{
#include <stdio.h>
%}

%define %scilab_asarray_withallocatecopy(TYPEMAPTYPE, FRAGMENTNAME, CTYPE, TEMPDATATYPE)
%typemap(TYPEMAPTYPE, fragment="FRAGMENTNAME") CTYPE {
  size_t i = 0;
  int iRows = 0;
  int iCols = 0;
  TEMPDATATYPE *pTempData = NULL;
  if (FRAGMENTNAME(pvApiCtx, $input, &iRows, &iCols, &pTempData, fname)) {
    return SWIG_ERROR;
  }
  $1 = ($1_ltype)MALLOC(sizeof($*1_ltype) * iRows * iCols);
  for (i = 0; i < iRows * iCols; i++) {
    $1[i] = ($*1_ltype) pTempData[i];
  }
}
%enddef

%define %scilab_asarrayandsize_withcopy(TYPEMAPTYPE, FRAGMENTNAME, CTYPE, TEMPDATATYPE)
%typemap(TYPEMAPTYPE, fragment="FRAGMENTNAME") CTYPE {
  int iRows = 0;
  int iCols = 0;
  TEMPDATATYPE *pTempData = NULL;
  if (FRAGMENTNAME(pvApiCtx, $input, &iRows, &iCols, &pTempData, fname)) {
    return SWIG_ERROR;
  }
  if (iRows*iCols <= $1_dim0) {
    size_t i;
    for (i = 0; i < $1_dim0; i++) {
      $1[i] = ($*1_ltype) pTempData[i];
    }
  }
  else {
    char errmsg[100];
    sprintf(errmsg, "Size of input data (%d) is too big (maximum is %d)",
      iRows*iCols, $1_dim0);
    SWIG_exception_fail(SWIG_OverflowError, errmsg);
  }
}
%enddef

%define %scilab_fromarrayandsize(TYPEMAPTYPE, FRAGMENTNAME, CTYPE)
%typemap(TYPEMAPTYPE, noblock=1, fragment="FRAGMENTNAME") CTYPE {
  %set_output(FRAGMENTNAME(pvApiCtx, $result, 1, $1_dim0, $1));
}
%enddef

%define %scilab_array_typemaps(CTYPE, ASARRAY_FRAGMENT, FROMARRAY_FRAGMENT, TEMPDATATYPE)
  %scilab_asarrayandsize_withcopy(varin, ASARRAY_FRAGMENT, CTYPE[ANY], TEMPDATATYPE);
  %scilab_asarray_withallocatecopy(in, ASARRAY_FRAGMENT, CTYPE[ANY], TEMPDATATYPE);
  %scilab_fromarrayandsize(varout, FROMARRAY_FRAGMENT, CTYPE[ANY]);
  %scilab_fromarrayandsize(out, FROMARRAY_FRAGMENT, CTYPE[ANY]);

  %apply SWIGTYPE[] { CTYPE[] };
  %scilab_asarray_withallocatecopy(in, ASARRAY_FRAGMENT, CTYPE[], TEMPDATATYPE);
%enddef


// Double
%scilab_array_typemaps(double, SWIG_SciDouble_AsDoubleArrayAndSize,
  SWIG_SciDouble_FromDoubleArrayAndSize, double);

// Signed char

%scilab_array_typemaps(signed char, SWIG_SciDoubleOrInt8_AsSignedCharArrayAndSize,
  SWIG_SciDouble_FromSignedCharArrayAndSize, signed char);

// Unsigned char
%scilab_array_typemaps(unsigned char, SWIG_SciDoubleOrUint8_AsUnsignedCharArrayAndSize,
  SWIG_SciDouble_FromUnsignedCharArrayAndSize, unsigned char);

// Short
%scilab_array_typemaps(short, SWIG_SciDoubleOrInt16_AsShortArrayAndSize,
  SWIG_SciDouble_FromShortArrayAndSize, short);

// Unsigned short
%scilab_array_typemaps(unsigned short, SWIG_SciDoubleOrUint16_AsUnsignedShortArrayAndSize,
  SWIG_SciDouble_FromUnsignedShortArrayAndSize, unsigned short);

// Int
%scilab_array_typemaps(int, SWIG_SciDoubleOrInt32_AsIntArrayAndSize,
  SWIG_SciDouble_FromIntArrayAndSize, int);

// Unsigned int
%scilab_array_typemaps(unsigned int, SWIG_SciDoubleOrUint32_AsUnsignedIntArrayAndSize,
  SWIG_SciDouble_FromUnsignedIntArrayAndSize, unsigned int);

// Long
%scilab_array_typemaps(long, SWIG_SciDoubleOrInt32_AsIntArrayAndSize,
  SWIG_SciDouble_FromLongArrayAndSize, int);

// Unsigned long
%scilab_array_typemaps(unsigned long, SWIG_SciDoubleOrUint32_AsUnsignedIntArrayAndSize,
  SWIG_SciDouble_FromUnsignedLongArrayAndSize, unsigned int);

// Float
%scilab_array_typemaps(float, SWIG_SciDouble_AsFloatArrayAndSize,
  SWIG_SciDouble_FromFloatArrayAndSize, float);

// Bool
%scilab_array_typemaps(bool, SWIG_SciBoolean_AsIntArrayAndSize,
  SWIG_SciBoolean_FromBoolArrayAndSize, int);

// Char *
%scilab_array_typemaps(char *, SWIG_SciString_AsCharPtrArrayAndSize,
  SWIG_SciString_FromCharPtrArrayAndSize, char *);

