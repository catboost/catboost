
/* ------------------------------------------------------------
 *  utility methods for char strings 
 * ------------------------------------------------------------ */
%fragment("SWIG_AsCharPtrAndSize", "header", fragment="SWIG_pchar_descriptor") {
SWIGINTERN int
SWIG_AsCharPtrAndSize(v8::Handle<v8::Value> valRef, char** cptr, size_t* psize, int *alloc)
{
  if(valRef->IsString()) {
    v8::Handle<v8::String> js_str = SWIGV8_TO_STRING(valRef);

    size_t len = SWIGV8_UTF8_LENGTH(js_str) + 1;
    char* cstr = new char[len];
    SWIGV8_WRITE_UTF8(js_str, cstr, len);
    
    if(alloc) *alloc = SWIG_NEWOBJ;
    if(psize) *psize = len;
    if(cptr) *cptr = cstr;
    
    return SWIG_OK;
  } else {
    if(valRef->IsObject()) {
      v8::Handle<v8::Object> obj = SWIGV8_TO_OBJECT(valRef);
      // try if the object is a wrapped char[]
      swig_type_info* pchar_descriptor = SWIG_pchar_descriptor();
      if (pchar_descriptor) {
        void* vptr = 0;
        if (SWIG_ConvertPtr(obj, &vptr, pchar_descriptor, 0) == SWIG_OK) {
          if (cptr) *cptr = (char *) vptr;
          if (psize) *psize = vptr ? (strlen((char *)vptr) + 1) : 0;
          if (alloc) *alloc = SWIG_OLDOBJ;
          return SWIG_OK;
        }
      }
      return SWIG_TypeError;
    } else {
      return SWIG_TypeError;
    }
  }
}
}

%fragment("SWIG_FromCharPtrAndSize","header",fragment="SWIG_pchar_descriptor") {
SWIGINTERNINLINE v8::Handle<v8::Value>
SWIG_FromCharPtrAndSize(const char* carray, size_t size)
{
  if (carray) {
    if (size > INT_MAX) {
      // TODO: handle extra long strings
      return SWIGV8_UNDEFINED();
    } else {
      v8::Handle<v8::String> js_str = SWIGV8_STRING_NEW2(carray, size);
      return js_str;
    }
  } else {
    return SWIGV8_UNDEFINED();
  }
}
}
