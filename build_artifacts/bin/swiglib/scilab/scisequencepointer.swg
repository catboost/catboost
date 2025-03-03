/*
 *
 * Scilab list of pointer <-> C++ pointer container
 *
 */

%include <scilist.swg>

%fragment("<stdint.h>", "header") {
%#include <stdint.h>
}

%fragment(SWIG_AsCheck_Sequence_frag(ptr), "header",
  fragment="SWIG_ScilabList") {

SWIGINTERN int
SWIG_AsCheck_Sequence_dec(ptr)(SwigSciObject obj) {
  return SWIG_CheckScilabList(obj);
}
}

%fragment(SWIG_AsGet_Sequence_frag(ptr), "header",
  fragment="SWIG_ScilabList") {

SWIGINTERN int
SWIG_AsGet_Sequence_dec(ptr)(SwigSciObject obj, int **piSequence) {
  return SWIG_GetScilabList(obj, piSequence);
}
}

%fragment(SWIG_AsSize_Sequence_frag(ptr), "header",
  fragment="SWIG_ScilabList") {

SWIGINTERN int
SWIG_AsSize_Sequence_dec(ptr)(SwigSciObject obj, int *piSize) {
  return SWIG_GetScilabListSize(obj, piSize);
}
}

%fragment(SWIG_FromCreate_Sequence_frag(ptr), "header",
  fragment="<stdint.h>") {

SWIGINTERN int
SWIG_FromCreate_Sequence_dec(ptr)(int size, uintptr_t **pSequence) {
  *pSequence = new uintptr_t[size];
  return *pSequence != NULL ? SWIG_OK : SWIG_ERROR;
}
}

%fragment(SWIG_FromSet_Sequence_frag(ptr), "header",
  fragment="<stdint.h>") {

SWIGINTERN SwigSciObject
SWIG_FromSet_Sequence_dec(ptr)(int size, uintptr_t *pSequence) {
  SciErr sciErr;
  int *piListAddr;

  int iVarOut = SWIG_NbInputArgument(pvApiCtx) + SWIG_Scilab_GetOutputPosition();

  sciErr = createList(pvApiCtx, iVarOut, size, &piListAddr);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }

  for (int i=0; i<size; i++) {
    sciErr = createPointerInList(pvApiCtx, iVarOut, piListAddr, i + 1, (void *)pSequence[i]);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return SWIG_ERROR;
    }
  }
  delete (int*)pSequence;
  return SWIG_OK;
}
}

%fragment(SWIG_AsVal_SequenceItem_frag(ptr), "header") {

SWIGINTERN void*
SWIG_AsVal_SequenceItem_dec(ptr)(SwigSciObject obj, int *piSequence, int itemIndex)
{
  SciErr sciErr;
  int *piItemAddr;
  int iType;
  void* pItemValue = NULL;

  sciErr = getListItemAddress(pvApiCtx, piSequence, itemIndex + 1, &piItemAddr);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return NULL;
  }

  sciErr = getVarType(pvApiCtx, piItemAddr, &iType);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return NULL;
  }

  if (iType != sci_pointer) {
    Scierror(SCILAB_API_ARGUMENT_ERROR, _("%s: Wrong type for input argument #%d: A pointer is expected at list item #%d.\n"), SWIG_Scilab_GetFuncName(), obj, itemIndex + 1);
    return NULL;
  }

  sciErr = getPointerInList(pvApiCtx, piSequence, itemIndex + 1, &pItemValue);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return NULL;
  }

  return pItemValue;
}
}

%fragment(SWIG_From_SequenceItem_frag(ptr), "header",
  fragment="<stdint.h>") {

SWIGINTERN int
SWIG_From_SequenceItem_dec(ptr)(uintptr_t *pSequence, int iItemIndex, uintptr_t itemValue) {
  pSequence[iItemIndex] = itemValue;
  return SWIG_OK;
}
}
