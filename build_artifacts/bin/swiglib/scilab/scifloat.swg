/*
 * FLOAT SCALAR
 */

%fragment(SWIG_AsVal_frag(float), "header", fragment=SWIG_AsVal_frag(double)) {
SWIGINTERN int
SWIG_AsVal_dec(float)(SwigSciObject iVar, float *pfValue) {
  double dblValue = 0.0;
  if(SWIG_AsVal_dec(double)(iVar, &dblValue) != SWIG_OK) {
    return SWIG_ERROR;
  }
  if (pfValue)
    *pfValue = (float) dblValue;
  return SWIG_OK;
}
}

%fragment(SWIG_From_frag(float), "header") {
SWIGINTERN int
SWIG_From_dec(float)(float flValue) {
  if (createScalarDouble(pvApiCtx, SWIG_NbInputArgument(pvApiCtx)
    + SWIG_Scilab_GetOutputPosition(), (double)flValue))
    return SWIG_ERROR;
  return SWIG_OK;
}
}

%fragment("SWIG_SciDouble_AsFloatArrayAndSize", "header") {
SWIGINTERN int
SWIG_SciDouble_AsFloatArrayAndSize(void *pvApiCtx, int iVar, int *iRows, int *iCols, float **pfValue, char *fname) {
  SciErr sciErr;
  int *piAddrVar = NULL;
  double *pdValue = NULL;

  sciErr = getVarAddressFromPosition(pvApiCtx, iVar, &piAddrVar);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  if (isDoubleType(pvApiCtx, piAddrVar) && !isVarComplex(pvApiCtx, piAddrVar)) {
    int i;

    sciErr = getMatrixOfDouble(pvApiCtx, piAddrVar, iRows, iCols, &pdValue);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }

    *pfValue = (float *) malloc((*iRows) * (*iCols) * sizeof(float));
    for (i=0; i < (*iRows) * (*iCols); i++)
      (*pfValue)[i] = (float) pdValue[i];

    return SWIG_OK;
  }
  else {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A real matrix expected.\n"), fname, iVar);
    return SWIG_ERROR;
  }
}
}

%fragment("SWIG_SciDouble_FromFloatArrayAndSize", "header") {
SWIGINTERN int
SWIG_SciDouble_FromFloatArrayAndSize(void *pvApiCtx, int iVarOut, int iRows, int iCols, float *pfValue) {
  SciErr sciErr;
  double *pdValue;
  int i;

  pdValue = (double *) malloc(iRows * iCols * sizeof(double));
  for (i = 0; i < iRows * iCols; i++)
    pdValue[i] = pfValue[i];

  sciErr = createMatrixOfDouble(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + iVarOut, iRows, iCols, pdValue);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  free(pdValue);
  return SWIG_OK;
}
}
