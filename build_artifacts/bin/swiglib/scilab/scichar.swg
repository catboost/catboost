/*
 * C-type: char or char*
 * Scilab type: string
 */

/*
 * CHAR
 */

%fragment(SWIG_AsVal_frag(char), "header", fragment="SWIG_SciString_AsChar") {
#define SWIG_AsVal_char(scilabValue, valuePointer) SWIG_SciString_AsChar(pvApiCtx, scilabValue, valuePointer, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_SciString_AsChar", "header") {
SWIGINTERN int
SWIG_SciString_AsChar(void *pvApiCtx, int iVar, char *pcValue, char *fname) {
  SciErr sciErr;
  int *piAddrVar = NULL;
  char *pstValue = NULL;
  int iRet;

  sciErr = getVarAddressFromPosition(pvApiCtx, iVar, &piAddrVar);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  if (isStringType(pvApiCtx, piAddrVar) == 0)
  {
      Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A single string expected.\n"), fname, iVar);
      return SWIG_TypeError;
  }

  iRet = getAllocatedSingleString(pvApiCtx, piAddrVar, &pstValue);
  if (iRet) {
    return SWIG_ERROR;
  }

  if (pcValue != NULL) {
    *pcValue = pstValue[0];
  }

  freeAllocatedSingleString(pstValue);
  return SWIG_OK;
}
}

%fragment(SWIG_From_frag(char), "header", fragment="SWIG_SciString_FromChar") {
#define SWIG_From_char(value) SWIG_SciString_FromChar(pvApiCtx, SWIG_Scilab_GetOutputPosition(), value)
}
%fragment("SWIG_SciString_FromChar", "header") {
SWIGINTERN int
SWIG_SciString_FromChar(void *pvApiCtx, int iVarOut, char chValue) {
  char *pchValue = (char*)malloc(sizeof(char) * 2);
  pchValue[0] = chValue;
  pchValue[1] = '\0';

  if (createSingleString(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + iVarOut, pchValue))
    return SWIG_ERROR;

  free(pchValue);
  return SWIG_OK;
}
}

/*
 * CHAR *
*/

%fragment("SWIG_AsCharArray", "header", fragment = "SWIG_SciString_AsCharPtr") {
#define SWIG_AsCharArray(scilabValue, charPtrPointer, charPtrLength) SWIG_SciString_AsCharPtr(pvApiCtx, scilabValue, charPtrPointer, charPtrLength, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_SciString_AsCharPtr", "header") {
SWIGINTERN int
SWIG_SciString_AsCharPtr(void *pvApiCtx, int iVar, char *pcValue, int iLength, char *fname) {
  SciErr sciErr;
  int *piAddrVar = NULL;
  char* pcTmpValue = NULL;
  int iRet;

  sciErr = getVarAddressFromPosition(pvApiCtx, iVar, &piAddrVar);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  iRet = getAllocatedSingleString(pvApiCtx, piAddrVar, &pcTmpValue);
  if (iRet) {
    return SWIG_ERROR;
  }

  if (pcValue != NULL) {
    strncpy(pcValue, pcTmpValue, iLength);
  }

  freeAllocatedSingleString(pcTmpValue);
  return SWIG_OK;
}
}

%fragment("SWIG_AsCharPtrAndSize", "header", fragment = "SWIG_SciString_AsCharPtrAndSize") {
#define SWIG_AsCharPtrAndSize(scilabValue, charPtrPointer, charPtrLength, allocMemory) SWIG_SciString_AsCharPtrAndSize(pvApiCtx, scilabValue, charPtrPointer, charPtrLength, allocMemory, SWIG_Scilab_GetFuncName())
}
%fragment("SWIG_SciString_AsCharPtrAndSize", "header") {
SWIGINTERN int
SWIG_SciString_AsCharPtrAndSize(void *pvApiCtx, int iVar, char **pcValue, size_t *piLength, int *alloc, char *fname) {
  SciErr sciErr;
  int *piAddrVar = NULL;
  char *pstString = NULL;
  int iRows = 0;
  int iCols = 0;
  int iLen = 0;

  sciErr = getVarAddressFromPosition(pvApiCtx, iVar, &piAddrVar);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  if (isScalar(pvApiCtx, piAddrVar) == 0 || isStringType(pvApiCtx, piAddrVar) == 0)
  {
      Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A single string expected.\n"), fname, iVar);
      return SWIG_TypeError;
  }

  sciErr = getMatrixOfString(pvApiCtx, piAddrVar, &iRows, &iCols, &iLen, NULL);
  if (sciErr.iErr)
  {
      printError(&sciErr, 0);
      return SWIG_ERROR;
  }

  pstString = %new_array(iLen + 1, char);

  sciErr = getMatrixOfString(pvApiCtx, piAddrVar, &iRows, &iCols, &iLen, &pstString);
  if (sciErr.iErr)
  {
      printError(&sciErr, 0);
      return SWIG_ERROR;
  }

  // TODO: return SWIG_ERROR if pcValue NULL (now returning SWIG_ERROR fails some typechecks)
  if (pcValue) {
    *pcValue = pstString;
  }

  if (alloc != NULL) {
    *alloc = SWIG_NEWOBJ;
  }

  if (piLength != NULL) {
    *piLength = strlen(pstString);
  }

  return SWIG_OK;
}
}

%fragment("SWIG_FromCharPtr", "header", fragment = "SWIG_SciString_FromCharPtr") {
#define SWIG_FromCharPtr(charPtr) SWIG_SciString_FromCharPtr(pvApiCtx, SWIG_Scilab_GetOutputPosition(), charPtr)
}
%fragment("SWIG_SciString_FromCharPtr", "header") {
SWIGINTERN int
SWIG_SciString_FromCharPtr(void *pvApiCtx, int iVarOut, const char *pchValue) {
  if (pchValue) {
    SciErr sciErr;
    const char* pstStrings[1];
    pstStrings[0] = pchValue;

    sciErr = createMatrixOfString(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + iVarOut, 1, 1, pstStrings);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }
  }
  else {
    int iRet = createEmptyMatrix(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + iVarOut);
    if (iRet) {
      return SWIG_ERROR;
    }
  }

  return SWIG_OK;
}
}

/*
 * CHAR * ARRAY
 */

%fragment("SWIG_SciString_AsCharPtrArrayAndSize", "header") {
SWIGINTERN int
SWIG_SciString_AsCharPtrArrayAndSize(void *pvApiCtx, int iVar, int *iRows, int *iCols, char ***charPtrArray, char *fname) {
  SciErr sciErr;
  int i = 0;
  int *piAddrVar = NULL;
  int* piLength = NULL;

  sciErr = getVarAddressFromPosition(pvApiCtx, iVar, &piAddrVar);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  sciErr = getMatrixOfString(pvApiCtx, piAddrVar, iRows, iCols, NULL, NULL);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  piLength = (int*) malloc((*iRows) * (*iCols) * sizeof(int));

  sciErr = getMatrixOfString(pvApiCtx, piAddrVar, iRows, iCols, piLength, NULL);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  *charPtrArray = (char**) malloc((*iRows) * (*iCols) * sizeof(char*));
  for(i = 0 ; i < (*iRows) * (*iCols); i++) {
    (*charPtrArray)[i] = (char*) malloc(sizeof(char) * (piLength[i] + 1));
  }

  sciErr = getMatrixOfString(pvApiCtx, piAddrVar, iRows, iCols, piLength, *charPtrArray);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  free(piLength);
  return SWIG_OK;
}
}

%fragment("SWIG_SciString_FromCharPtrArrayAndSize", "header") {
SWIGINTERN int
SWIG_SciString_FromCharPtrArrayAndSize(void *pvApiCtx, int iVarOut, int iRows, int iCols, char **charPtrArray) {
  SciErr sciErr;

  sciErr = createMatrixOfString(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + iVarOut, iRows, iCols, (const char* const*) charPtrArray);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  return SWIG_OK;
}
}

%fragment("SWIG_FromCharPtrAndSize", "header", fragment = "SWIG_SciString_FromCharPtr") {
#define SWIG_FromCharPtrAndSize(charPtr, charPtrLength) SWIG_SciString_FromCharPtr(pvApiCtx, SWIG_Scilab_GetOutputPosition(), charPtr)
}


/*
 * Char* Scilab variable
 */

%fragment(SWIG_CreateScilabVariable_frag(char), "wrapper") {
SWIGINTERN int
SWIG_CreateScilabVariable_dec(char)(void *pvApiCtx, const char* psVariableName, const char cVariableValue) {
  SciErr sciErr;
  char sValue[2];
  const char* psStrings[1];

  sValue[0] = cVariableValue;
  sValue[1] = '\0';
  psStrings[0] = sValue;

  sciErr = createNamedMatrixOfString(pvApiCtx, psVariableName, 1, 1, psStrings);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }
  return SWIG_OK;
}
}

%fragment(SWIG_CreateScilabVariable_frag(charptr), "wrapper") {
SWIGINTERN int
SWIG_CreateScilabVariable_dec(charptr)(void *pvApiCtx, const char* psVariableName, const char* psVariableValue) {
  SciErr sciErr;
  const char* psStrings[1];
  psStrings[0] = psVariableValue;

  sciErr = createNamedMatrixOfString(pvApiCtx, psVariableName, 1, 1, psStrings);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }
  return SWIG_OK;
}
}
