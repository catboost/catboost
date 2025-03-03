/*
 *
 * Scilab matrix of bool <-> C++ bool container
 *
 */

%include <scibool.swg>

%fragment(SWIG_AsCheck_Sequence_frag(bool), "header") {

SWIGINTERN int
SWIG_AsCheck_Sequence_dec(bool)(SwigSciObject obj) {
  SciErr sciErr;
  int *piAddrVar;

  sciErr = getVarAddressFromPosition(pvApiCtx, obj, &piAddrVar);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  if (isBooleanType(pvApiCtx, piAddrVar)) {
    return SWIG_OK;
  }
  else {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A boolean is expected.\n"), SWIG_Scilab_GetFuncName(), obj);
    return SWIG_ERROR;
  }
}
}

%fragment(SWIG_AsGet_Sequence_frag(bool), "header",
  fragment="SWIG_SciBoolean_AsIntArrayAndSize") {

SWIGINTERN int
SWIG_AsGet_Sequence_dec(bool)(SwigSciObject obj, int **pSequence) {
  int iMatrixRowCount;
  int iMatrixColCount;
  return (SWIG_SciBoolean_AsIntArrayAndSize(pvApiCtx, obj, &iMatrixRowCount, &iMatrixColCount, pSequence, SWIG_Scilab_GetFuncName()));
}
}

%fragment(SWIG_AsSize_Sequence_frag(bool), "header",
  fragment="SWIG_SciBoolean_AsIntArrayAndSize") {

SWIGINTERN int
SWIG_AsSize_Sequence_dec(bool)(SwigSciObject obj, int *piSize) {
  int *piMatrix;
  int iMatrixRowCount;
  int iMatrixColCount;
  if (SWIG_SciBoolean_AsIntArrayAndSize(pvApiCtx, obj, &iMatrixRowCount, &iMatrixColCount, &piMatrix, SWIG_Scilab_GetFuncName()) == SWIG_OK) {
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

%fragment(SWIG_FromCreate_Sequence_frag(bool), "header") {

SWIGINTERN int
SWIG_FromCreate_Sequence_dec(bool)(int size, int **pSequence) {
  *pSequence = new int[size];
  return *pSequence != NULL ? SWIG_OK : SWIG_ERROR;
}
}

%fragment(SWIG_FromSet_Sequence_frag(bool), "header",
  fragment="SWIG_SciBoolean_FromIntArrayAndSize") {

SWIGINTERN SwigSciObject
SWIG_FromSet_Sequence_dec(bool)(int size, int *pSequence) {
  SwigSciObject obj = SWIG_SciBoolean_FromIntArrayAndSize(pvApiCtx, SWIG_Scilab_GetOutputPosition(), 1, size, pSequence);
  delete (int *)pSequence;
  return obj;
}
}

%fragment(SWIG_AsVal_SequenceItem_frag(bool), "header") {

SWIGINTERN bool
SWIG_AsVal_SequenceItem_dec(bool)(SwigSciObject obj, int *pSequence, int iItemIndex) {
  return (bool) pSequence[iItemIndex];
}
}

%fragment(SWIG_From_SequenceItem_frag(bool), "header") {

SWIGINTERN int
SWIG_From_SequenceItem_dec(bool)(int *pSequence, int iItemIndex, bool itemValue) {
  pSequence[iItemIndex] = itemValue;
  return SWIG_OK;
}
}
