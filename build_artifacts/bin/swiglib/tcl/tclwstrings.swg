/* -----------------------------------------------------------------------------
 * tclwstrings.wg
 *
 * Utility methods for wchar strings 
 * ----------------------------------------------------------------------------- */

%{
#include <wchar.h>
%}

%fragment("SWIG_AsWCharPtrAndSize","header") {
SWIGINTERN int
SWIG_AsWCharPtrAndSize(Tcl_Obj *obj, wchar_t** cptr, size_t* psize, int *alloc)
{ 
  int len = 0;
  Tcl_UniChar *ustr = Tcl_GetUnicodeFromObj(obj, &len);
  if (ustr) {
    if (cptr)  {
      Tcl_Encoding encoding = NULL;
      char *src = (char *) ustr;
      int srcLen = (len)*sizeof(Tcl_UniChar);
      int dstLen = sizeof(wchar_t)*(len + 1);
      char *dst = %new_array(dstLen, char);
      int flags = 0;
      Tcl_EncodingState *statePtr = 0;
      int srcRead = 0;
      int dstWrote = 0;
      int dstChars = 0;
      Tcl_UtfToExternal(0, encoding, src, srcLen, flags, statePtr, dst,
			dstLen, &srcRead, &dstWrote, &dstChars);
      
      if (alloc) *alloc = SWIG_NEWOBJ;
    }
    if (psize) *psize = len + 1;
    return SWIG_OK;
  }
  return SWIG_TypeError;
}
}

%fragment("SWIG_FromWCharPtrAndSize","header") {
SWIGINTERNINLINE Tcl_Obj *
SWIG_FromWCharPtrAndSize(const wchar_t* carray, size_t size)
{
  Tcl_Obj *res = NULL;
  if (size < INT_MAX) {
    Tcl_Encoding encoding = NULL;
    char *src = (char *) carray;
    int srcLen = (int)(size*sizeof(wchar_t));
    int dstLen = (int)(size*sizeof(Tcl_UniChar));
    char *dst = %new_array(dstLen, char);
    int flags = 0;
    Tcl_EncodingState *statePtr = 0;
    int srcRead = 0;
    int dstWrote = 0;
    int dstChars = 0;
    
    Tcl_ExternalToUtf(0, encoding, src, srcLen, flags, statePtr, dst,
		      dstLen, &srcRead, &dstWrote, &dstChars);
    
    res = Tcl_NewUnicodeObj((Tcl_UniChar*)dst, (int)size);
    %delete_array(dst);
  }
  return res;
}
}

