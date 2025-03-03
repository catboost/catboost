/*
 * C-type: unsigned long
 * Scilab type: see unsigned int
 */

%fragment(SWIG_AsVal_frag(unsigned long), "header", fragment="SWIG_UnsignedInt_AsUnsignedLong") {
#define SWIG_AsVal_unsigned_SS_long(scilabValue, valuePointer) SWIG_UnsignedInt_AsUnsignedLong(pvApiCtx, scilabValue, valuePointer, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_UnsignedInt_AsUnsignedLong", "header", fragment=SWIG_AsVal_frag(unsigned int)) {
SWIGINTERN int
SWIG_UnsignedInt_AsUnsignedLong(void *pvApiCtx, SwigSciObject iVar, unsigned long *pulValue, char *fname) {
  unsigned int uiValue = 0;
  if(SWIG_AsVal_unsigned_SS_int(iVar, &uiValue) != SWIG_OK) {
    return SWIG_ERROR;
  }
  *pulValue = (unsigned long) uiValue;
  return SWIG_OK;
}
}

%fragment(SWIG_From_frag(unsigned long), "header", fragment="SWIG_UnsignedInt_FromUnsignedLong") {
#define SWIG_From_unsigned_SS_long(scilabValue) SWIG_UnsignedInt_FromUnsignedLong(pvApiCtx, SWIG_Scilab_GetOutputPosition(), scilabValue, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_UnsignedInt_FromUnsignedLong", "header", fragment=SWIG_From_frag(unsigned int)) {
SWIGINTERN int
SWIG_UnsignedInt_FromUnsignedLong(void *pvApiCtx, int iVarOut, unsigned long ulValue, char *fname) {
  return SWIG_From_unsigned_SS_int((unsigned int)ulValue);
}
}

%fragment("SWIG_SciDouble_FromUnsignedLongArrayAndSize", "header") {
SWIGINTERN int
SWIG_SciDouble_FromUnsignedLongArrayAndSize(void *pvApiCtx, int iVarOut, int iRows, int iCols, const unsigned long *pulValues) {
  SciErr sciErr;
  double *pdValues = NULL;
  int i;

  pdValues = (double*) malloc(iRows * iCols * sizeof(double));
  for (i=0; i<iRows * iCols; i++) {
    pdValues[i] = pulValues[i];
  }

  sciErr = createMatrixOfDouble(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + iVarOut, iRows, iCols, pdValues);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    free(pdValues);
    return SWIG_ERROR;
  }

  free(pdValues);
  return SWIG_OK;
}
}
