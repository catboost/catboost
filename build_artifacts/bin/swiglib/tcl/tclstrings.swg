/* ------------------------------------------------------------
 *  utility methods for char strings 
 * ------------------------------------------------------------ */

%fragment("SWIG_AsCharPtrAndSize","header") {
SWIGINTERN int
SWIG_AsCharPtrAndSize(Tcl_Obj *obj, char** cptr, size_t* psize, int *alloc)
{ 
  int len = 0;
  char *cstr = Tcl_GetStringFromObj(obj, &len);
  if (cstr) {
    if (cptr)  *cptr = cstr;
    if (psize) *psize = len + 1;
    if (alloc) *alloc = SWIG_OLDOBJ;
    return SWIG_OK;
  }
  return SWIG_TypeError;
}
}


%fragment("SWIG_FromCharPtrAndSize","header",
	  fragment="<limits.h>") {
SWIGINTERNINLINE Tcl_Obj *
SWIG_FromCharPtrAndSize(const char* carray, size_t size)
{
  return (size < INT_MAX) ? Tcl_NewStringObj(carray, %numeric_cast(size,int)) : NULL;
}
}


