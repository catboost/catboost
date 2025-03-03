/*
 * C-type: int
 * Scilab type: double or int32
 */

%fragment(SWIG_AsVal_frag(int), "header", fragment="SWIG_SciDoubleOrInt32_AsInt", fragment="<limits.h>") {
%#define SWIG_AsVal_int(scilabValue, valuePointer) SWIG_SciDoubleOrInt32_AsInt(pvApiCtx, scilabValue, valuePointer, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_SciDoubleOrInt32_AsInt", "header") {
SWIGINTERN int
SWIG_SciDoubleOrInt32_AsInt(void *pvApiCtx, SwigSciObject iVar, int *piValue, char *fname)
{
  SciErr sciErr;
  int iType = 0;
  int iRows = 0;
  int iCols = 0;
  int *piAddrVar = NULL;

  sciErr = getVarAddressFromPosition(pvApiCtx, iVar, &piAddrVar);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  sciErr = getVarType(pvApiCtx, piAddrVar, &iType);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  if (iType == sci_ints) {
    if (piValue) {
      int iPrec = 0;
      int *piData = NULL;

      sciErr = getMatrixOfIntegerPrecision(pvApiCtx, piAddrVar, &iPrec);
      if (sciErr.iErr) {
        printError(&sciErr, 0);
        return SWIG_ERROR;
      }
      if (iPrec != SCI_INT32) {
        Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A 32-bit signed integer or a double expected.\n"), fname, iVar);
        return SWIG_TypeError;
      }
      sciErr = getMatrixOfInteger32(pvApiCtx, piAddrVar, &iRows, &iCols, &piData);
      if (sciErr.iErr) {
        printError(&sciErr, 0);
        return SWIG_ERROR;
      }
      if (iRows * iCols != 1) {
        Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong size for input argument #%d: A 32-bit signed integer or a double expected.\n"), fname, iVar);
        return SWIG_TypeError;
      }
      *piValue = *piData;
    }
  }
  else if (iType == sci_matrix) {
    if (piValue) {
      double *pdData = NULL;
      double dValue = 0.0f;
      sciErr = getMatrixOfDouble(pvApiCtx, piAddrVar, &iRows, &iCols, &pdData);
      if (sciErr.iErr) {
        printError(&sciErr, 0);
        return SWIG_ERROR;
      }
      if (iRows * iCols != 1) {
        Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong size for input argument #%d: A 32-bit signed integer or a double expected.\n"), fname, iVar);
        return SWIG_TypeError;
      }
      dValue = *pdData;
      if (dValue != floor(dValue)) {
        Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Incorrect value for input argument #%d: The double value cannot be converted to a 32-bit signed integer.\n"), fname, iVar);
        return SWIG_ValueError;
      }
      if ((dValue < INT_MIN) || (dValue > INT_MAX)) {
        Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Overflow error for input argument #%d: The double value cannot be converted to a 32-bit signed integer.\n"), fname, iVar);
        return SWIG_OverflowError;
      }
      *piValue = (int) dValue;
    }
  }
  else {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A 32-bit signed integer or a double expected.\n"), fname, iVar);
    return SWIG_TypeError;
  }

  return SWIG_OK;
}
}

%fragment(SWIG_From_frag(int), "header", fragment="SWIG_SciDouble_FromInt") {
%#define SWIG_From_int(scilabValue) SWIG_SciDouble_FromInt(pvApiCtx, SWIG_Scilab_GetOutputPosition(), scilabValue, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_SciDouble_FromInt", "header") {
SWIGINTERN int
SWIG_SciDouble_FromInt(void *pvApiCtx, int iVarOut, int iValue, char *fname){
  if (createScalarDouble(pvApiCtx, SWIG_NbInputArgument(pvApiCtx)
    + iVarOut, (double) iValue))
    return SWIG_ERROR;
  return SWIG_OK;
}
}

/*
 * C-type: int[]
 * Scilab type: double or int32 matrix
 */
%fragment("SWIG_SciDoubleOrInt32_AsIntArrayAndSize", "header") {
SWIGINTERN int
SWIG_SciDoubleOrInt32_AsIntArrayAndSize(void *pvApiCtx, int iVar, int *iRows, int *iCols, int **piValue, char *fname) {
  SciErr sciErr;
  int iType = 0;
  int *piAddrVar = NULL;

  sciErr = getVarAddressFromPosition(pvApiCtx, iVar, &piAddrVar);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  sciErr = getVarType(pvApiCtx, piAddrVar, &iType);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  if (iType == sci_matrix) {
    double *pdData = NULL;
    int size = 0;
    int i;

    sciErr = getMatrixOfDouble(pvApiCtx, piAddrVar, iRows, iCols, &pdData);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }

    size = (*iRows) * (*iCols);
    *piValue = (int*) malloc(size * sizeof(int*));
    for (i = 0; i < size; i++)
      (*piValue)[i] = (int) pdData[i];
  }
  else if (iType == sci_ints) {
    int iPrec = 0;
    sciErr = getMatrixOfIntegerPrecision(pvApiCtx, piAddrVar, &iPrec);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }
    if (iPrec != SCI_INT32) {
      Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A 32-bit signed integer or a double matrix expected.\n"), fname, iVar);
      return SWIG_ERROR;
    }
    sciErr = getMatrixOfInteger32(pvApiCtx, piAddrVar, iRows, iCols, piValue);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }
  }
  else {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A 32-bit signed integer or a double matrix expected.\n"), fname, iVar);
    return SWIG_ERROR;
  }
  return SWIG_OK;
}
}

%fragment("SWIG_SciDouble_FromIntArrayAndSize", "header") {
SWIGINTERN int
SWIG_SciDouble_FromIntArrayAndSize(void *pvApiCtx, int iVarOut, int iRows, int iCols, const int *piData) {
  SciErr sciErr;
  double *pdValues = NULL;
  int i;

  pdValues = (double*) malloc(iRows * iCols * sizeof(double));
  for (i=0; i<iRows * iCols; i++)
    pdValues[i] = piData[i];

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

%fragment(SWIG_CreateScilabVariable_frag(int), "wrapper") {
SWIGINTERN int
SWIG_CreateScilabVariable_dec(int)(void *pvApiCtx, const char* psVariableName, const int iVariableValue) {
  SciErr sciErr;
  sciErr = createNamedMatrixOfInteger32(pvApiCtx, psVariableName, 1, 1, &iVariableValue);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }
  return SWIG_OK;
}
}
