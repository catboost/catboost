/* ------------------------------------------------------------
 *  utility methods for char strings 
 * ------------------------------------------------------------ */

%fragment("SWIG_AsCharPtrAndSize","header",fragment="SWIG_pchar_descriptor") {
SWIGINTERN int
SWIG_AsCharPtrAndSize(VALUE obj, char** cptr, size_t* psize, int *alloc)
{
  if (TYPE(obj) == T_STRING) {
    char *cstr = StringValuePtr(obj); 
    size_t size = RSTRING_LEN(obj) + 1;
    if (cptr)  {
      if (alloc) {
	if (*alloc == SWIG_NEWOBJ) {
	  *cptr = %new_copy_array(cstr, size, char);
	} else {
	  *cptr = cstr;
	  *alloc = SWIG_OLDOBJ;
	}
      }
    }
    if (psize) *psize = size;
    return SWIG_OK;
  } else {
    swig_type_info* pchar_descriptor = SWIG_pchar_descriptor();
    if (pchar_descriptor) {
      void* vptr = 0;
      if (SWIG_ConvertPtr(obj, &vptr, pchar_descriptor, 0) == SWIG_OK) {
	if (cptr) *cptr = (char *)vptr;
	if (psize) *psize = vptr ? (strlen((char*)vptr) + 1) : 0;
	if (alloc) *alloc = SWIG_OLDOBJ;
	return SWIG_OK;
      }
    }
  }  
  return SWIG_TypeError;
}
}

%fragment("SWIG_FromCharPtrAndSize","header",fragment="SWIG_pchar_descriptor") {
SWIGINTERNINLINE VALUE 
SWIG_FromCharPtrAndSize(const char* carray, size_t size)
{
  if (carray) {
    if (size > LONG_MAX) {
      swig_type_info* pchar_descriptor = SWIG_pchar_descriptor();
      return pchar_descriptor ? 
	SWIG_NewPointerObj(%const_cast(carray,char *), pchar_descriptor, 0) : Qnil;
    } else {
      return rb_str_new(carray, %numeric_cast(size,long));
    }
  } else {
    return Qnil;
  }
}
}

