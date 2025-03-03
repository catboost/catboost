/*
 *
 * Scilab matrix of double <-> C++ double container
 *
 */

%include <scidouble.swg>

%fragment(SWIG_AsCheck_Sequence_frag(double), "header") {

SWIGINTERN int
SWIG_AsCheck_Sequence_dec(double)(SwigSciObject obj) {
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

%fragment(SWIG_AsGet_Sequence_frag(double), "header",
  fragment="SWIG_SciDouble_AsDoubleArrayAndSize") {

SWIGINTERN int
SWIG_AsGet_Sequence_dec(double)(SwigSciObject obj, double **pSequence) {
  int iMatrixRowCount;
  int iMatrixColCount;
  return (SWIG_SciDouble_AsDoubleArrayAndSize(pvApiCtx, obj, &iMatrixRowCount, &iMatrixColCount, pSequence, SWIG_Scilab_GetFuncName()));
}
}

%fragment(SWIG_AsSize_Sequence_frag(double), "header",
  fragment="SWIG_SciDouble_AsDoubleArrayAndSize") {

SWIGINTERN int
SWIG_AsSize_Sequence_dec(double)(SwigSciObject obj, int *piSize) {
  double *pdblMatrix;
  int iMatrixRowCount;
  int iMatrixColCount;
  if (SWIG_SciDouble_AsDoubleArrayAndSize(pvApiCtx, obj, &iMatrixRowCount, &iMatrixColCount, &pdblMatrix, SWIG_Scilab_GetFuncName()) == SWIG_OK) {
    if ((iMatrixRowCount > 1) && (iMatrixColCount > 1)) {
      Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong size for input argument #%d: A double vector is expected.\n"), SWIG_Scilab_GetFuncName(), obj);
      return SWIG_ERROR;
    }
    *piSize = iMatrixRowCount * iMatrixColCount;
    return SWIG_OK;
  }
  return SWIG_ERROR;
}
}

%fragment(SWIG_FromCreate_Sequence_frag(double), "header") {

SWIGINTERN int
SWIG_FromCreate_Sequence_dec(double)(int size, double **pSequence) {
  *pSequence = new double[size];
  return *pSequence != NULL ? SWIG_OK : SWIG_ERROR;
}
}

%fragment(SWIG_FromSet_Sequence_frag(double), "header",
  fragment="SWIG_SciDouble_FromDoubleArrayAndSize") {

SWIGINTERN SwigSciObject
SWIG_FromSet_Sequence_dec(double)(int size, double *pSequence) {
  SwigSciObject obj = SWIG_SciDouble_FromDoubleArrayAndSize(pvApiCtx, SWIG_Scilab_GetOutputPosition(), 1, size, pSequence);
  delete (double *)pSequence;
  return obj;
}
}

%fragment(SWIG_AsVal_SequenceItem_frag(double), "header") {

SWIGINTERN double
SWIG_AsVal_SequenceItem_dec(double)(SwigSciObject obj, double *pSequence, int iItemIndex) {
  return pSequence[iItemIndex];
}
}

%fragment(SWIG_From_SequenceItem_frag(double), "header") {

SWIGINTERN int
SWIG_From_SequenceItem_dec(double)(double *pSequence, int iItemIndex, double itemValue) {
  pSequence[iItemIndex] = itemValue;
  return SWIG_OK;
}
}

