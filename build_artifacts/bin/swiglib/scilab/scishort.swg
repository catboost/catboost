/*
 * C-type: short
 * Scilab type: double or int16
 */

%fragment(SWIG_AsVal_frag(short), "header", fragment="SWIG_SciDoubleOrInt16_AsShort", fragment="<limits.h>") {
#define SWIG_AsVal_short(scilabValue, valuePointer) SWIG_SciDoubleOrInt16_AsShort(pvApiCtx, scilabValue, valuePointer, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_SciDoubleOrInt16_AsShort", "header") {
SWIGINTERN int
SWIG_SciDoubleOrInt16_AsShort(void *pvApiCtx, int iVar, short *psValue, char *fname) {
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
    int iPrec = 0;
    short *psData = NULL;

    sciErr = getMatrixOfIntegerPrecision(pvApiCtx, piAddrVar, &iPrec);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }
    if (iPrec != SCI_INT16) {
      Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A 16-bit signed integer or a double expected.\n"), fname, iVar);
      return SWIG_TypeError;
    }
    sciErr = getMatrixOfInteger16(pvApiCtx, piAddrVar, &iRows, &iCols, &psData);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }
    if (iRows * iCols != 1) {
      Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong size for input argument #%d: A 16-bit signed integer or a double expected.\n"), fname, iVar);
      return SWIG_TypeError;
    }
    *psValue = *psData;
  }
  else if (iType == sci_matrix) {
    double *pdData = NULL;
    double dValue = 0.0f;

    sciErr = getMatrixOfDouble(pvApiCtx, piAddrVar, &iRows, &iCols, &pdData);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }
    if (iRows * iCols != 1) {
      Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong size for input argument #%d: A 16-bit signed integer or a double expected.\n"), fname, iVar);
      return SWIG_TypeError;
    }
    dValue = *pdData;
    if (dValue != floor(dValue)) {
      Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Incorrect value for input argument #%d: The double value cannot be converted to a 16-bit signed integer.\n"), fname, iVar);
      return SWIG_ValueError;
    }
    if ((dValue < SHRT_MIN) || (dValue > SHRT_MAX)) {
      Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Overflow error for input argument #%d: The double value cannot be converted to a 16-bit signed integer.\n"), fname, iVar);
      return SWIG_OverflowError;
    }
    *psValue = (short) dValue;
  }
  else {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A 16-bit signed integer or a double expected.\n"), fname, iVar);
    return SWIG_TypeError;
  }

  return SWIG_OK;
}
}

%fragment(SWIG_From_frag(short), "header", fragment="SWIG_SciDouble_FromShort") {
#define SWIG_From_short(scilabValue) SWIG_SciDouble_FromShort(pvApiCtx, SWIG_Scilab_GetOutputPosition(), scilabValue, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_SciDouble_FromShort", "header") {
SWIGINTERN int
SWIG_SciDouble_FromShort(void *pvApiCtx, int iVarOut, short sValue, char *fname) {
  if (createScalarDouble(pvApiCtx,
    SWIG_NbInputArgument(pvApiCtx) + iVarOut, (double) sValue))
    return SWIG_ERROR;
  return SWIG_OK;
}
}

/*
 * C-type: short[]
 * Scilab type: double or int16 matrix
 */
%fragment("SWIG_SciDoubleOrInt16_AsShortArrayAndSize", "header") {
SWIGINTERN int
SWIG_SciDoubleOrInt16_AsShortArrayAndSize(void *pvApiCtx, int iVar, int *iRows, int *iCols, short **psValue, char *fname) {
  SciErr sciErr;
  int iType = 0;
  int iPrec = 0;
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
    *psValue = (short*) malloc(size * sizeof(int*));
    for (i = 0; i < size; i++)
      (*psValue)[i] = (short) pdData[i];
  }
  else if (iType == sci_ints) {
    int iPrec = 0;
    sciErr = getMatrixOfIntegerPrecision(pvApiCtx, piAddrVar, &iPrec);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }
    if (iPrec != SCI_INT16) {
      Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A 16-bit signed integer or a double matrix expected.\n"), fname, iVar);
      return SWIG_ERROR;
    }

    sciErr = getMatrixOfInteger16(pvApiCtx, piAddrVar, iRows, iCols, psValue);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }
  }
  else {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A 16-bit signed integer or a double matrix expected.\n"), fname, iVar);
    return SWIG_ERROR;
  }

  return SWIG_OK;
}
}
%fragment("SWIG_SciDouble_FromShortArrayAndSize", "header") {
SWIGINTERN int
SWIG_SciDouble_FromShortArrayAndSize(void *pvApiCtx, int iVarOut, int iRows, int iCols, short *psValue) {
  SciErr sciErr;
  int i;
  double *pdValues = NULL;

  pdValues = (double*) malloc(iRows * iCols * sizeof(double));
  for (i=0; i<iRows * iCols; i++)
    pdValues[i] = psValue[i];

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

