/*
 *
 * Scilab matrix of int <-> C++ int container
 *
 */

%include <sciint.swg>

%fragment(SWIG_AsCheck_Sequence_frag(int), "header") {

SWIGINTERN int
SWIG_AsCheck_Sequence_dec(int)(SwigSciObject obj) {
  SciErr sciErr;
  int *piAddrVar;
  int iType = 0;

  sciErr = getVarAddressFromPosition(pvApiCtx, obj, &piAddrVar);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  sciErr = getVarType(pvApiCtx, piAddrVar, &iType);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  if ((iType == sci_matrix) || (iType == sci_ints)) {
    return SWIG_OK;
  }
  else {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: An integer is expected.\n"), SWIG_Scilab_GetFuncName(), obj);
    return SWIG_ERROR;
  }
}
}

%fragment(SWIG_AsGet_Sequence_frag(int), "header",
  fragment="SWIG_SciDoubleOrInt32_AsIntArrayAndSize") {
SWIGINTERN int
SWIG_AsGet_Sequence_dec(int)(SwigSciObject obj, int **pSequence) {
  int iMatrixRowCount;
  int iMatrixColCount;
  return (SWIG_SciDoubleOrInt32_AsIntArrayAndSize(pvApiCtx, obj, &iMatrixRowCount, &iMatrixColCount, pSequence, SWIG_Scilab_GetFuncName()));
}
}

%fragment(SWIG_AsSize_Sequence_frag(int), "header",
  fragment="SWIG_SciDoubleOrInt32_AsIntArrayAndSize") {

SWIGINTERN int
SWIG_AsSize_Sequence_dec(int)(SwigSciObject obj, int *piSize) {
  int *piMatrix;
  int iMatrixRowCount;
  int iMatrixColCount;
  if (SWIG_SciDoubleOrInt32_AsIntArrayAndSize(pvApiCtx, obj, &iMatrixRowCount, &iMatrixColCount, &piMatrix, SWIG_Scilab_GetFuncName()) == SWIG_OK) {
    if ((iMatrixRowCount > 1) && (iMatrixColCount > 1)) {
      Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong size for input argument #%d: An integer vector is expected.\n"), SWIG_Scilab_GetFuncName(), obj);
      return SWIG_ERROR;
    }
    *piSize = iMatrixRowCount * iMatrixColCount;
    return SWIG_OK;
  }
  return SWIG_ERROR;
}
}

%fragment(SWIG_FromCreate_Sequence_frag(int), "header") {

SWIGINTERN int
SWIG_FromCreate_Sequence_dec(int)(int size, int **pSequence) {
  *pSequence = new int[size];
  return *pSequence != NULL ? SWIG_OK : SWIG_ERROR;
}
}

%fragment(SWIG_FromSet_Sequence_frag(int), "header",
  fragment="SWIG_SciDouble_FromIntArrayAndSize") {

SWIGINTERN SwigSciObject
SWIG_FromSet_Sequence_dec(int)(int size, int *pSequence) {
  SwigSciObject obj = SWIG_SciDouble_FromIntArrayAndSize(pvApiCtx, SWIG_Scilab_GetOutputPosition(), 1, size, pSequence);
  delete (int *)pSequence;
  return obj;
}
}

%fragment(SWIG_AsVal_SequenceItem_frag(int), "header") {

SWIGINTERN int
SWIG_AsVal_SequenceItem_dec(int)(SwigSciObject obj, int *pSequence, int iItemIndex) {
  return pSequence[iItemIndex];
}
}

%fragment(SWIG_From_SequenceItem_frag(int), "header") {

SWIGINTERN int
SWIG_From_SequenceItem_dec(int)(int *pSequence, int iItemIndex, int itemValue) {
  pSequence[iItemIndex] = itemValue;
  return SWIG_OK;
}
}
