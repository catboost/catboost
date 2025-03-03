/*
 *
 * Scilab matrix of string <-> C++ std::string container
 *
 */

%include <scichar.swg>

%fragment(SWIG_AsCheck_Sequence_frag(std::string), "header") {

SWIGINTERN int
SWIG_AsCheck_Sequence_dec(std::string)(SwigSciObject obj) {
  SciErr sciErr;
  int *piAddrVar;

  sciErr = getVarAddressFromPosition(pvApiCtx, obj, &piAddrVar);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  if (isStringType(pvApiCtx, piAddrVar)) {
    return SWIG_OK;
  }
  else {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A string is expected.\n"), SWIG_Scilab_GetFuncName(), obj);
    return SWIG_ERROR;
  }
}
}

%fragment(SWIG_AsGet_Sequence_frag(std::string), "header",
  fragment="SWIG_SciString_AsCharPtrArrayAndSize") {

SWIGINTERN int
SWIG_AsGet_Sequence_dec(std::string)(SwigSciObject obj, char ***pSequence) {
  int iRows = 0;
  int iCols = 0;
  return (SWIG_SciString_AsCharPtrArrayAndSize(pvApiCtx, obj, &iRows, &iCols, pSequence, SWIG_Scilab_GetFuncName()));
}
}

%fragment(SWIG_AsSize_Sequence_frag(std::string), "header",
  fragment="SWIG_SciString_AsCharPtrArrayAndSize") {

SWIGINTERN int
SWIG_AsSize_Sequence_dec(std::string)(SwigSciObject obj, int *piSize) {
  char **pstMatrix;
  int iCols = 0;
  int iRows = 0;
  if (SWIG_SciString_AsCharPtrArrayAndSize(pvApiCtx, obj, &iRows, &iCols, &pstMatrix, SWIG_Scilab_GetFuncName()) == SWIG_OK) {
    *piSize = iRows * iCols;
    return SWIG_OK;
  }
  return SWIG_ERROR;
}
}

%fragment(SWIG_FromCreate_Sequence_frag(std::string), "header") {

SWIGINTERN int
SWIG_FromCreate_Sequence_dec(std::string)(int size, char ***pSequence) {
  *pSequence = new char*[size];
  return *pSequence != NULL ? SWIG_OK : SWIG_ERROR;
}
}

%fragment(SWIG_FromSet_Sequence_frag(std::string), "header",
  fragment="SWIG_SciString_FromCharPtrArrayAndSize") {

SWIGINTERN SwigSciObject
SWIG_FromSet_Sequence_dec(std::string)(int size, char **pSequence) {
  SwigSciObject obj = SWIG_SciString_FromCharPtrArrayAndSize(pvApiCtx, SWIG_Scilab_GetOutputPosition(), 1, size, pSequence);
  delete (char **)pSequence;
  return obj;
}
}

%fragment(SWIG_AsVal_SequenceItem_frag(std::string), "header") {

SWIGINTERN std::string
SWIG_AsVal_SequenceItem_dec(std::string)(SwigSciObject obj, char **pSequence, int iItemIndex) {
  return std::string(pSequence[iItemIndex]);
}
}

%fragment(SWIG_From_SequenceItem_frag(std::string), "header") {

SWIGINTERN int
SWIG_From_SequenceItem_dec(std::string)(char **pSequence, int iItemIndex, std::string itemValue) {
  char *pChar = new char((int) itemValue.size() + 1);
  strcpy(pChar, itemValue.c_str());
  pSequence[iItemIndex] = pChar;
  return SWIG_OK;
}
}

