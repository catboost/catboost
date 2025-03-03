/*
 * DOUBLE SCALAR
 */
%fragment(SWIG_AsVal_frag(double), "header", fragment="SWIG_SciDouble_AsDouble") {
%#define SWIG_AsVal_double(scilabValue, valuePointer) SWIG_SciDouble_AsDouble(pvApiCtx, scilabValue, valuePointer, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_SciDouble_AsDouble", "header") {
SWIGINTERN int
SWIG_SciDouble_AsDouble(void *pvApiCtx, SwigSciObject iVar, double *pdblValue, char *fname) {
  SciErr sciErr;
  int iRet = 0;
  int *piAddrVar = NULL;

  sciErr = getVarAddressFromPosition(pvApiCtx, iVar, &piAddrVar);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  if (!isDoubleType(pvApiCtx, piAddrVar) || isVarComplex(pvApiCtx, piAddrVar)) {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A real expected.\n"), fname, iVar);
    return SWIG_ERROR;
  }

  if (!isScalar(pvApiCtx, piAddrVar)) {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong size for input argument #%d: A real expected.\n"), fname, iVar);
    return SWIG_ERROR;
  }

  iRet = getScalarDouble(pvApiCtx, piAddrVar, pdblValue);
  if (iRet) {
    return SWIG_ERROR;
  }

  return SWIG_OK;
}
}

%fragment(SWIG_From_frag(double), "header", fragment="SWIG_SciDouble_FromDouble") { 
%#define SWIG_From_double(scilabValue) SWIG_SciDouble_FromDouble(pvApiCtx, SWIG_Scilab_GetOutputPosition(), scilabValue, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_SciDouble_FromDouble", "header") {
SWIGINTERN int
SWIG_SciDouble_FromDouble(void *pvApiCtx, int iVarOut, double dblValue, char *fname) {
  if (createScalarDouble(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + iVarOut, dblValue))
    return SWIG_ERROR;
  return SWIG_OK;
}
}

/*
 * DOUBLE ARRAY
 */

%fragment("SWIG_SciDouble_AsDoubleArrayAndSize", "header") {
SWIGINTERN int
SWIG_SciDouble_AsDoubleArrayAndSize(void *pvApiCtx, int iVar, int *iRows, int *iCols, double **pdValue, char *fname) {
  SciErr sciErr;
  int *piAddrVar = NULL;

  sciErr = getVarAddressFromPosition(pvApiCtx, iVar, &piAddrVar);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  if (isDoubleType(pvApiCtx, piAddrVar) && !isVarComplex(pvApiCtx, piAddrVar)) {
    sciErr = getMatrixOfDouble(pvApiCtx, piAddrVar, iRows, iCols, pdValue);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }
  }
  else {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A real matrix expected.\n"), fname, iVar);
    return SWIG_ERROR;
  }

  return SWIG_OK;
}
}

%fragment("SWIG_SciDouble_FromDoubleArrayAndSize", "header") {
SWIGINTERN int
SWIG_SciDouble_FromDoubleArrayAndSize(void *pvApiCtx, int iVarOut, int iRows, int iCols, double *pdblValue) {
  SciErr sciErr;
  sciErr = createMatrixOfDouble(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + iVarOut, iRows, iCols, pdblValue);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  return SWIG_OK;
}
}

%fragment(SWIG_CreateScilabVariable_frag(double), "wrapper") {
SWIGINTERN int
SWIG_CreateScilabVariable_dec(double)(void *pvApiCtx, const char* psVariableName, const double dVariableValue) {
  SciErr sciErr;
  sciErr = createNamedMatrixOfDouble(pvApiCtx, psVariableName, 1, 1, &dVariableValue);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }
  return SWIG_OK;
}
}
