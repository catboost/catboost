/*
 * C-type: unsigned int
 * Scilab type: double or uint32
 */
%fragment(SWIG_AsVal_frag(unsigned int), "header", fragment="SWIG_SciDoubleOrUint32_AsUnsignedInt", fragment="<limits.h>") {
%#define SWIG_AsVal_unsigned_SS_int(scilabValue, valuePointer) SWIG_SciDoubleOrUint32_AsUnsignedInt(pvApiCtx, scilabValue, valuePointer, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_SciDoubleOrUint32_AsUnsignedInt", "header") {
SWIGINTERN int
SWIG_SciDoubleOrUint32_AsUnsignedInt(void *pvApiCtx, int iVar, unsigned int *puiValue, char *fname) {
  SciErr sciErr;
  int iType = 0;
  int iRows = 0;
  int iCols = 0;
  int iPrec = 0;
  int *piAddrVar = NULL;
  unsigned int *puiData = NULL;

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
    if (puiValue) {
      sciErr = getMatrixOfIntegerPrecision(pvApiCtx, piAddrVar, &iPrec);
      if (sciErr.iErr) {
        printError(&sciErr, 0);
        return SWIG_ERROR;
      }
      if (iPrec != SCI_UINT32) {
        Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A 32-bit unsigned integer or a double expected.\n"), fname, iVar);
        return SWIG_ERROR;
      }

      sciErr = getMatrixOfUnsignedInteger32(pvApiCtx, piAddrVar, &iRows, &iCols, &puiData);
      if (sciErr.iErr) {
        printError(&sciErr, 0);
        return SWIG_ERROR;
      }
      if (iRows * iCols != 1) {
        Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong size for input argument #%d: A 32-bit unsigned integer or a double expected.\n"), fname, iVar);
        return SWIG_ERROR;
      }
      *puiValue = *puiData;
    }
  }
  else if (iType == sci_matrix) {
    if (puiValue) {
      double *pdData = NULL;
      double dValue = 0.0f;
      sciErr = getMatrixOfDouble(pvApiCtx, piAddrVar, &iRows, &iCols, &pdData);
      if (sciErr.iErr) {
        printError(&sciErr, 0);
        return SWIG_ERROR;
      }
      if (iRows * iCols != 1) {
        Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong size for input argument #%d: A 32-bit unsigned integer or a double expected.\n"), fname, iVar);
        return SWIG_TypeError;
      }
      dValue = *pdData;
      if (dValue != floor(dValue)) {
        Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Incorrect value for input argument #%d: The double value cannot be converted to a 32-bit unsigned integer.\n"), fname, iVar);
        return SWIG_ValueError;
      }
      if ((dValue < 0) || (dValue > UINT_MAX)) {
        Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Overflow error for input argument #%d: The double value cannot be converted to a 32-bit unsigned integer.\n"), fname, iVar);
        return SWIG_OverflowError;
      }
      *puiValue = (unsigned int) dValue;
    }
  }
  else {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A 32-bit unsigned integer or a double expected.\n"), fname, iVar);
    return SWIG_ERROR;
  }

  return SWIG_OK;
}
}

%fragment(SWIG_From_frag(unsigned int), "header", fragment="SWIG_SciDouble_FromUnsignedInt") {
%#define SWIG_From_unsigned_SS_int(scilabValue) SWIG_SciDouble_FromUnsignedInt(pvApiCtx, SWIG_Scilab_GetOutputPosition(), scilabValue, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_SciDouble_FromUnsignedInt", "header") {
SWIGINTERN int
SWIG_SciDouble_FromUnsignedInt(void *pvApiCtx, int iVarOut, unsigned int uiValue, char *fname) {
  if (createScalarDouble(pvApiCtx,
    SWIG_NbInputArgument(pvApiCtx) + iVarOut, (double) uiValue))
    return SWIG_ERROR;
  return SWIG_OK;
}
}

/*
 * C-type: unsigned int[]
 * Scilab type: uint32 vector
 */
%fragment("SWIG_SciDoubleOrUint32_AsUnsignedIntArrayAndSize", "header") {
SWIGINTERN int
SWIG_SciDoubleOrUint32_AsUnsignedIntArrayAndSize(void *pvApiCtx, int iVar, int *iRows, int *iCols, unsigned int **puiValue, char *fname) {
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
    *puiValue = (unsigned int*) malloc(size * sizeof(int*));
    for (i = 0; i < size; i++)
      (*puiValue)[i] = (unsigned int) pdData[i];
  }
  else if (iType == sci_ints) {
    sciErr = getMatrixOfIntegerPrecision(pvApiCtx, piAddrVar, &iPrec);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }

    if (iPrec != SCI_UINT32) {
      Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A 32-bit unsigned integer or a double vector expected.\n"), fname, iVar);
      return SWIG_ERROR;
    }

    sciErr = getMatrixOfUnsignedInteger32(pvApiCtx, piAddrVar, iRows, iCols, puiValue);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }
  }
  else {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A 32-bit unsigned integer or a double vector expected.\n"), fname, iVar);
    return SWIG_ERROR;
  }

  return SWIG_OK;
}
}

%fragment("SWIG_SciDouble_FromUnsignedIntArrayAndSize", "header") {
SWIGINTERN int
SWIG_SciDouble_FromUnsignedIntArrayAndSize(void *pvApiCtx, int iVarOut, int iRows, int iCols, unsigned int *puiValues) {
  SciErr sciErr;
  double *pdValues = NULL;
  int i;

  pdValues = (double*) malloc(iRows * iCols * sizeof(double));
  for (i=0; i<iRows * iCols; i++)
    pdValues[i] = puiValues[i];

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

%fragment(SWIG_CreateScilabVariable_frag(uint), "wrapper") {
SWIGINTERN int
SWIG_CreateScilabVariable_dec(uint)(void *pvApiCtx, const char* psVariableName, const unsigned int uiVariableValue) {
  SciErr sciErr;
  sciErr = createNamedMatrixOfUnsignedInteger32(pvApiCtx, psVariableName, 1, 1, &uiVariableValue);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }
  return SWIG_OK;
}
}

