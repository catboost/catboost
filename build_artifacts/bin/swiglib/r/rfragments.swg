/* for raw pointers */
#define SWIG_ConvertPtr(oc, ptr, ty, flags)             SWIG_R_ConvertPtr(oc, ptr, ty, flags)
#define SWIG_ConvertFunctionPtr(oc, ptr, ty)            SWIG_R_ConvertPtr(oc, ptr, ty, 0)
#define SWIG_NewPointerObj(ptr, ty, flags)              SWIG_R_NewPointerObj(ptr, ty, flags)
#define SWIG_NewFunctionPtrObj(ptr, ty)                 SWIG_R_NewPointerObj(ptr, ty, 0)

/* for raw packed data */
#define SWIG_ConvertPacked(obj, ptr, sz, ty)            SWIG_R_ConvertPacked(obj, ptr, sz, ty)
#define SWIG_NewPackedObj(ptr, sz, ty)                  SWIG_R_NewPackedObj(ptr, sz, ty)

/* for class or struct pointers */
#define SWIG_ConvertInstance(obj, pptr, ty, flags)      SWIG_ConvertPtr(obj, pptr, ty, flags)
#define SWIG_NewInstanceObj(ptr, ty, flags)             SWIG_NewPointerObj(ptr, ty, flags)

/* for C++ member pointers, ie, member methods */
#define SWIG_ConvertMember(obj, ptr, sz, ty)            SWIG_R_ConvertPacked(obj, ptr, sz, ty)
#define SWIG_NewMemberObj(ptr, sz, ty)                  SWIG_R_NewPackedObj(ptr, sz, ty)


/* Runtime API */

#define SWIG_GetModule(clientdata)                      SWIG_R_GetModule()
#define SWIG_SetModule(clientdata, pointer)             SWIG_R_SetModule(pointer)

%fragment(SWIG_From_frag(long),"header") {
SWIGINTERNINLINE SEXP
SWIG_From_dec(long)(long value)
{
	return Rf_ScalarInteger((int)value);
}
}

%fragment(SWIG_AsVal_frag(long),"header") {
SWIGINTERNINLINE  int
SWIG_AsVal_dec(long)(SEXP obj, long *val)
{
   if (val) *val = Rf_asInteger(obj);
   return SWIG_OK;
}
}


%fragment(SWIG_From_frag(long long),"header",
	  fragment="SWIG_LongLongAvailable") {
%#ifdef SWIG_LONG_LONG_AVAILABLE
SWIGINTERNINLINE SEXP
SWIG_From_dec(long long)(long long value)
{
	return Rf_ScalarInteger((int)value);
}
%#endif
}

%fragment(SWIG_AsVal_frag(long long),"header",
	  fragment="SWIG_LongLongAvailable") {
%#ifdef SWIG_LONG_LONG_AVAILABLE
SWIGINTERNINLINE  int
SWIG_AsVal_dec(long long)(SEXP obj, long long *val)
{
   if (val) *val = Rf_asInteger(obj);
   return SWIG_OK;
}
%#endif
}

%fragment(SWIG_From_frag(unsigned long),"header") {
SWIGINTERNINLINE SEXP
SWIG_From_dec(unsigned long)(unsigned long value)
{
	return Rf_ScalarInteger((int)value);
}
}


%fragment(SWIG_AsVal_frag(unsigned long),"header") {
SWIGINTERNINLINE  int
SWIG_AsVal_dec(unsigned long)(SEXP obj, unsigned long *val)
{
   if (val) *val = Rf_asInteger(obj);
   return SWIG_OK;
}
}


%fragment(SWIG_From_frag(unsigned long long),"header",
	  fragment="SWIG_LongLongAvailable") {
%#ifdef SWIG_LONG_LONG_AVAILABLE
SWIGINTERNINLINE SEXP
SWIG_From_dec(unsigned long long)(unsigned long long value)
{
	return Rf_ScalarInteger((int)value);
}
%#endif
}


%fragment(SWIG_AsVal_frag(unsigned long long),"header",
	  fragment="SWIG_LongLongAvailable") {
%#ifdef SWIG_LONG_LONG_AVAILABLE
SWIGINTERNINLINE  int
SWIG_AsVal_dec(unsigned long long)(SEXP obj, unsigned long long *val)
{
   if (val) *val = Rf_asInteger(obj);
   return SWIG_OK;
}
%#endif
}

%fragment(SWIG_From_frag(double),"header") {
SWIGINTERNINLINE SEXP
SWIG_From_dec(double)(double value)
{
	return Rf_ScalarReal(value);
}
}


%fragment(SWIG_AsVal_frag(double),"header") {
SWIGINTERNINLINE  int
SWIG_AsVal_dec(double)(SEXP obj, double *val)
{
   if (val) *val = Rf_asReal(obj);
   return SWIG_OK;
}
}

%fragment("SWIG_AsCharPtrAndSize", "header") 
{
SWIGINTERN int
SWIG_AsCharPtrAndSize(SEXP obj, char** cptr, size_t* psize, int *alloc)
{
  if (cptr && Rf_isString(obj)) {
    char *cstr = %const_cast(CHAR(STRING_ELT(obj, 0)), char *);
    int len = strlen(cstr);

    if (alloc) {
      if (*alloc == SWIG_NEWOBJ) {
        *cptr = %new_copy_array(cstr, len + 1, char);
        *alloc = SWIG_NEWOBJ;
      } else {
        *cptr = cstr;
      }
    } else {
      *cptr = %reinterpret_cast(malloc(len + 1), char *);
      *cptr = strcpy(*cptr, cstr);
    }
    if (psize) *psize = len + 1;
    return SWIG_OK;
  }
  return SWIG_TypeError;
}
}

%fragment("SWIG_strdup","header") 
{
SWIGINTERN char *
SWIG_strdup(const char *str)
{
  char *newstr = %reinterpret_cast(malloc(strlen(str) + 1), char *);
  return strcpy(newstr, str);
}
}

//# This is modified from the R header files

%fragment("SWIG_FromCharPtrAndSize","header") 
{
SWIGINTERN SEXP
SWIG_FromCharPtrAndSize(const char* carray, size_t size) 
{
  SEXP t, c;
  if (!carray) return R_NilValue;
/* See R internals document 1.10.  
   MkCharLen was introduced in 2.7.0.  Use that instead of hand
   creating vector.

   Starting in 2.8.0 creating strings via vectors was deprecated in
   order to allow for use of CHARSXP caches. */

  Rf_protect(t = Rf_allocVector(STRSXP, 1));
%#if R_VERSION >=  R_Version(2,7,0)
  c = Rf_mkCharLen(carray, size);
%#else
  c = Rf_allocVector(CHARSXP, size);
  strncpy((char *)CHAR(c), carray, size);
%#endif
  SET_STRING_ELT(t, 0, c);
  Rf_unprotect(1);
  return t;
}
}
