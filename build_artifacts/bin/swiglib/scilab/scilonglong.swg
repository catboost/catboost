/*
 * C-type: long long
 * Scilab 5 type: NONE
 * Scilab 6 type: int64
 */
%fragment(SWIG_AsVal_frag(long long), "header", fragment="SWIG_SciInt64_ToLongLong") {
%#define SWIG_AsVal_long_SS_long(scilabValue, valuePointer) SWIG_SciInt64_ToLongLong(pvApiCtx, scilabValue, valuePointer, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_SciInt64_ToLongLong", "header") {
SWIGINTERN int
SWIG_SciInt64_ToLongLong(void *pvApiCtx, int iVar, long long *pllValue, char *fname) {
  Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Scilab 5.X does not manage '%s' data type.\n"), "SWIG_SciInt64_ToLongLong", "int64");
  return SWIG_ERROR;
}
}

%fragment(SWIG_From_frag(long long), "header", fragment="SWIG_SciInt64_FromLongLong") {
#define SWIG_From_long_SS_long(value) SWIG_SciInt64_FromLongLong(pvApiCtx, $result, value)
}
%fragment("SWIG_SciInt64_FromLongLong", "header") {
SWIGINTERN int
SWIG_SciInt64_FromLongLong(void *pvApiCtx, int iVarOut, long long llValue) {
  Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Scilab 5.X does not manage '%s' data type.\n"), "SWIG_SciInt64_ToLongLong", "int64");
  return SWIG_ERROR;
}
}

/*
 * C-type: unsigned long long
 * Scilab 5 type: NONE
 * Scilab 6 type: uint64
 */
%fragment(SWIG_AsVal_frag(unsigned long long), "header", fragment="SWIG_SciUint64_ToUnsignedLongLong") {
#define SWIG_AsVal_unsigned_SS_long_SS_long(scilabValue, valuePointer) SWIG_SciUint64_ToUnsignedLongLong(pvApiCtx, scilabValue, valuePointer, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_SciUint64_ToUnsignedLongLong", "header") {
SWIGINTERN int
SWIG_SciUint64_ToUnsignedLongLong(void *pvApiCtx, int iVar, unsigned long long *pullValue, char *fname) {
  Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Scilab 5.X does not manage '%s' data type.\n"), "SWIG_SciUint64_ToLongLong", "uint64");
  return SWIG_ERROR;
}
}

%fragment(SWIG_From_frag(unsigned long long), "header", fragment="SWIG_SciUint64_FromUnsignedLongLong") {
#define SWIG_From_unsigned_SS_long_SS_long(value) SWIG_SciUint64_FromUnsignedLongLong(pvApiCtx, $result, value)
}
%fragment("SWIG_SciUint64_FromUnsignedLongLong", "header") {
SWIGINTERN int
SWIG_SciUint64_FromUnsignedLongLong(void *pvApiCtx, int iVarOut, unsigned long long llValue) {
  Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Scilab 5.X does not manage '%s' data type.\n"), "SWIG_SciUint64_ToLongLong", "uint64");
  return SWIG_ERROR;
}
}

