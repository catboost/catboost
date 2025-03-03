/*
 *
 * Scilab matrix of float <-> C++ float container
 *
 */

%include <scifloat.swg>

%fragment(SWIG_AsCheck_Sequence_frag(float), "header") {

SWIGINTERN int
SWIG_AsCheck_Sequence_dec(float)(SwigSciObject obj) {
  SciErr sciErr;
  int *piAddrVar;

  sciErr = getVarAddressFromPosition(pvApiCtx, obj, &piAddrVar);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  if (isDoubleType(pvApiCtx, piAddrVar)) {
    return SWIG_OK;
  }
  else {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A double is expected.\n"), SWIG_Scilab_GetFuncName(), obj);
    return SWIG_ERROR;
  }
}
}

%fragment(SWIG_AsGet_Sequence_frag(float), "header",
  fragment="SWIG_SciDouble_AsFloatArrayAndSize") {

SWIGINTERN int
SWIG_AsGet_Sequence_dec(float)(SwigSciObject obj, float **pSequence) {
  int iMatrixRowCount;
  int iMatrixColCount;
  return (SWIG_SciDouble_AsFloatArrayAndSize(pvApiCtx, obj, &iMatrixRowCount, &iMatrixColCount, pSequence, SWIG_Scilab_GetFuncName()));
}
}

%fragment(SWIG_AsSize_Sequence_frag(float), "header",
  fragment="SWIG_SciDouble_AsFloatArrayAndSize") {

SWIGINTERN int
SWIG_AsSize_Sequence_dec(float)(SwigSciObject obj, int *piSize) {
  float *pdblMatrix;
  int iMatrixRowCount;
  int iMatrixColCount;
  if (SWIG_SciDouble_AsFloatArrayAndSize(pvApiCtx, obj, &iMatrixRowCount, &iMatrixColCount, &pdblMatrix, SWIG_Scilab_GetFuncName()) == SWIG_OK) {
    if ((iMatrixRowCount > 1) && (iMatrixColCount > 1)) {
      Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong size for input argument #%d: A float vector is expected.\n"), SWIG_Scilab_GetFuncName(), obj);
      return SWIG_ERROR;
    }
    *piSize = iMatrixRowCount * iMatrixColCount;
    return SWIG_OK;
  }
  return SWIG_ERROR;
}
}

%fragment(SWIG_FromCreate_Sequence_frag(float), "header") {

SWIGINTERN int
SWIG_FromCreate_Sequence_dec(float)(int size, float **pSequence) {
  *pSequence = new float[size];
  return *pSequence != NULL ? SWIG_OK : SWIG_ERROR;
}
}

%fragment(SWIG_FromSet_Sequence_frag(float), "header",
  fragment="SWIG_SciDouble_FromFloatArrayAndSize") {

SWIGINTERN SwigSciObject
SWIG_FromSet_Sequence_dec(float)(int size, float *pSequence) {
  SwigSciObject obj = SWIG_SciDouble_FromFloatArrayAndSize(pvApiCtx, SWIG_Scilab_GetOutputPosition(), 1, size, pSequence);
  delete (float *)pSequence;
  return obj;
}
}

%fragment(SWIG_AsVal_SequenceItem_frag(float), "header") {

SWIGINTERN float
SWIG_AsVal_SequenceItem_dec(float)(SwigSciObject obj, float *pSequence, int iItemIndex) {
  return pSequence[iItemIndex];
}
}

%fragment(SWIG_From_SequenceItem_frag(float), "header") {
SWIGINTERN int
SWIG_From_SequenceItem_dec(float)(float *pSequence, int iItemIndex, float itemValue) {
  pSequence[iItemIndex] = itemValue;
  return SWIG_OK;
}
}

