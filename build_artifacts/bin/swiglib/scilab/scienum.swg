/*
 * C-type: enum
 * Scilab type: double or int32
 */

%fragment(SWIG_AsVal_frag(Enum), "header", fragment="SWIG_Int_AsEnum") {
%#define SWIG_AsVal_Enum(scilabValue, valuePointer) SWIG_Int_AsEnum(pvApiCtx, scilabValue, valuePointer, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_Int_AsEnum", "header", fragment="SWIG_SciDoubleOrInt32_AsInt") {
SWIGINTERN int
SWIG_Int_AsEnum(void *pvApiCtx, int iVar, int *enumValue, char *fname) {
  int iValue = 0;
  if (SWIG_SciDoubleOrInt32_AsInt(pvApiCtx, iVar, &iValue, fname) != SWIG_OK)
    return SWIG_ERROR;
  *enumValue = iValue;
  return SWIG_OK;
}
}

%fragment(SWIG_From_frag(Enum), "header", fragment="SWIG_Int_FromEnum") {
%#define SWIG_From_Enum(scilabValue) SWIG_Int_FromEnum(pvApiCtx, SWIG_Scilab_GetOutputPosition(), scilabValue, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_Int_FromEnum", "header", fragment="SWIG_SciDouble_FromInt") {
SWIGINTERN int
SWIG_Int_FromEnum(void *pvApiCtx, int iVarOut, int enumValue, char *fname) {
  if (SWIG_SciDouble_FromInt(pvApiCtx, iVarOut, enumValue, fname) != SWIG_OK)
    return SWIG_ERROR;
  SWIG_Scilab_SetOutput(pvApiCtx, iVarOut);
  return SWIG_OK;
}
}
