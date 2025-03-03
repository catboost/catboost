// Other primitive such as size_t and ptrdiff_t

/*
 * C-type: size_t
 * Scilab type: double or int32
 */

%fragment(SWIG_AsVal_frag(size_t), "header", fragment="SWIG_Int_AsSize") {
%#define SWIG_AsVal_size_t(scilabValue, valuePointer) SWIG_Int_AsSize(pvApiCtx, scilabValue, valuePointer, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_Int_AsSize", "header", fragment=SWIG_AsVal_frag(int))
{
SWIGINTERN int
SWIG_Int_AsSize(void *pvApiCtx, SwigSciObject iVar, size_t *piValue, char *fname) {
  int iValue = 0;
  if (SWIG_AsVal_dec(int)(iVar, &iValue) != SWIG_OK)
    return SWIG_ERROR;

  if (piValue)
    *piValue = (size_t) iValue;

  return SWIG_OK;
}
}

%fragment(SWIG_From_frag(size_t), "header", fragment="SWIG_Int_FromSize") {
%#define SWIG_From_size_t(scilabValue) SWIG_Int_FromSize(pvApiCtx, SWIG_Scilab_GetOutputPosition(), scilabValue, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_Int_FromSize", "header", fragment=SWIG_From_frag(int))
{
SWIGINTERN int
SWIG_Int_FromSize(void *pvApiCtx, int iVarOut, size_t iValue, char *fname) {
  return SWIG_From_dec(int)((int)iValue);
}
}

/*
 * C-type: ptrdiff_t
 * Scilab type: double or int32
 */

%fragment(SWIG_AsVal_frag(ptrdiff_t), "header", fragment="SWIG_Int_AsPtrDiff") {
%#define SWIG_AsVal_ptrdiff_t(scilabValue, valuePointer) SWIG_Int_AsPtrDiff(pvApiCtx, scilabValue, valuePointer, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_Int_AsPtrDiff", "header", fragment=SWIG_AsVal_frag(int))
{
SWIGINTERN int
SWIG_Int_AsPtrDiff(void *pvApiCtx, SwigSciObject iVar, ptrdiff_t *piValue, char *fname) {
  int iValue = 0;
  if (SWIG_AsVal_dec(int)(iVar, &iValue) != SWIG_OK)
    return SWIG_ERROR;

  if (piValue)
    *piValue = (ptrdiff_t) iValue;

  return SWIG_OK;
}
}

%fragment(SWIG_From_frag(ptrdiff_t), "header", fragment="SWIG_Int_FromPtrDiff") {
%#define SWIG_From_ptrdiff_t(scilabValue) SWIG_Int_FromPtrDiff(pvApiCtx, SWIG_Scilab_GetOutputPosition(), scilabValue, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_Int_FromPtrDiff", "header", fragment=SWIG_From_frag(int)) {
SWIGINTERN int
SWIG_Int_FromPtrDiff(void *pvApiCtx, int iVarOut, ptrdiff_t iValue, char *fname) {
  return SWIG_From_dec(int)((int)iValue);
}
}

