/*
 * C-type: unsigned short
 * Scilab type: double or uint16
 */
%fragment(SWIG_AsVal_frag(unsigned short), "header", fragment="SWIG_SciDoubleOrUint16_AsUnsignedShort", fragment="<limits.h>") {
%#define SWIG_AsVal_unsigned_SS_short(scilabValue, valuePointer) SWIG_SciDoubleOrUint16_AsUnsignedShort(pvApiCtx, scilabValue, valuePointer, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_SciDoubleOrUint16_AsUnsignedShort", "header") {
SWIGINTERN int
SWIG_SciDoubleOrUint16_AsUnsignedShort(void *pvApiCtx, int iVar, unsigned short *pusValue, char *fname) {
  SciErr sciErr;
  int iType = 0;
  int iRows = 0;
  int iCols = 0;
  int iPrec = 0;
  int *piAddrVar = NULL;
  unsigned short *pusData = NULL;

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
    if (pusValue) {
      sciErr = getMatrixOfIntegerPrecision(pvApiCtx, piAddrVar, &iPrec);
      if (sciErr.iErr) {
        printError(&sciErr, 0);
        return SWIG_ERROR;
      }
      if (iPrec != SCI_UINT16) {
        Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A 16-bit unsigned integer or a double expected.\n"), fname, iVar);
        return SWIG_ERROR;
      }

      sciErr = getMatrixOfUnsignedInteger16(pvApiCtx, piAddrVar, &iRows, &iCols, &pusData);
      if (sciErr.iErr) {
        printError(&sciErr, 0);
        return SWIG_ERROR;
      }
      if (iRows * iCols != 1) {
        Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong size for input argument #%d: A 16-bit unsigned integer or a double expected.\n"), fname, iVar);
        return SWIG_ERROR;
      }
      *pusValue = *pusData;
    }
  }
  else if (iType == sci_matrix) {
    if (pusValue) {
      double *pdData = NULL;
      double dValue = 0.0f;
      sciErr = getMatrixOfDouble(pvApiCtx, piAddrVar, &iRows, &iCols, &pdData);
      if (sciErr.iErr) {
        printError(&sciErr, 0);
        return SWIG_ERROR;
      }
      if (iRows * iCols != 1) {
        Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong size for input argument #%d: A 16-bit unsigned integer or a double expected.\n"), fname, iVar);
        return SWIG_TypeError;
      }
      dValue = *pdData;
      if (dValue != floor(dValue)) {
        Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Incorrect value for input argument #%d: The double value cannot be converted to a 16-bit unsigned integer.\n"), fname, iVar);
        return SWIG_ValueError;
      }
      if ((dValue < 0) || (dValue > USHRT_MAX)) {
        Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Overflow error for input argument #%d: The double value cannot be converted to a 16-bit unsigned integer.\n"), fname, iVar);
        return SWIG_OverflowError;
      }
      *pusValue = (unsigned short) dValue;
    }
  }
  else {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A 16-bit unsigned integer or a double expected.\n"), fname, iVar);
    return SWIG_ERROR;
  }

  return SWIG_OK;
}
}

%fragment(SWIG_From_frag(unsigned short), "header", fragment="SWIG_SciDouble_FromUnsignedShort") {
%#define SWIG_From_unsigned_SS_short(scilabValue) SWIG_SciDouble_FromUnsignedShort(pvApiCtx, SWIG_Scilab_GetOutputPosition(), scilabValue, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_SciDouble_FromUnsignedShort", "header") {
SWIGINTERN int
SWIG_SciDouble_FromUnsignedShort(void *pvApiCtx, int iVarOut, unsigned short usValue, char *fname) {
  if (createScalarDouble(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + iVarOut, (double) usValue))
    return SWIG_ERROR;
  return SWIG_OK;
}
}

/*
 * C-type: unsigned short[]
 * Scilab type: uint16 vector
 */
%fragment("SWIG_SciDoubleOrUint16_AsUnsignedShortArrayAndSize", "header") {
SWIGINTERN int
SWIG_SciDoubleOrUint16_AsUnsignedShortArrayAndSize(void *pvApiCtx, int iVar, int *iRows, int *iCols, unsigned short **pusValue, char *fname) {
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
    *pusValue = (unsigned short*) malloc(size * sizeof(int*));
    for (i = 0; i < size; i++)
      (*pusValue)[i] = (unsigned short) pdData[i];
  }
  else if (iType == sci_ints) {
    sciErr = getMatrixOfIntegerPrecision(pvApiCtx, piAddrVar, &iPrec);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }

    if (iPrec != SCI_UINT16) {
      Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A 16-bit unsigned integer or a double vector expected.\n"), fname, iVar);
      return SWIG_ERROR;
    }

    sciErr = getMatrixOfUnsignedInteger16(pvApiCtx, piAddrVar, iRows, iCols, pusValue);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }
  }
  else {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A 16-bit unsigned integer or a double vector expected.\n"), fname, iVar);
    return SWIG_ERROR;
  }

  return SWIG_OK;
}
}

%fragment("SWIG_SciDouble_FromUnsignedShortArrayAndSize", "header") {
SWIGINTERN int
SWIG_SciDouble_FromUnsignedShortArrayAndSize(void *pvApiCtx, int iVarOut, int iRows, int iCols, unsigned short *pusValues) {
  SciErr sciErr;
  double *pdValues = NULL;
  int i;

  pdValues = (double*) malloc(iRows * iCols * sizeof(double));
  for (i=0; i<iRows * iCols; i++)
    pdValues[i] = pusValues[i];

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
