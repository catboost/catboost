/*
 * Scilab list related functions
 *
 */

%fragment("SWIG_ScilabList", "header")
{
SWIGINTERN int
SWIG_GetScilabList(SwigSciObject obj, int **piListAddr)
{
  SciErr sciErr;

  sciErr = getVarAddressFromPosition(pvApiCtx, obj, piListAddr);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  return SWIG_OK;
}

SWIGINTERN int
SWIG_GetScilabListSize(SwigSciObject obj, int *piListSize)
{
  SciErr sciErr;
  int *piListAddr;

  sciErr = getVarAddressFromPosition(pvApiCtx, obj, &piListAddr);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  sciErr = getListItemNumber(pvApiCtx, piListAddr, piListSize);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  return SWIG_OK;
}

SWIGINTERN int
SWIG_GetScilabListAndSize(SwigSciObject obj, int **piListAddr, int *piListSize)
{
  SciErr sciErr;

  sciErr = getVarAddressFromPosition(pvApiCtx, obj, piListAddr);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  sciErr = getListItemNumber(pvApiCtx, *piListAddr, piListSize);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  return SWIG_OK;
}

SWIGINTERN int
SWIG_CheckScilabList(SwigSciObject obj)
{
  SciErr sciErr;
  int *piListAddr;
  int iType;

  sciErr = getVarAddressFromPosition(pvApiCtx, obj, &piListAddr);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  sciErr = getVarType(pvApiCtx, piListAddr, &iType);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  if ((iType != sci_list) && (iType != sci_tlist) && (iType != sci_mlist)) {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A list is expected.\n"), SWIG_Scilab_GetFuncName(), obj);
    return SWIG_ERROR;
  }

  return SWIG_OK;
}

}

