/* -----------------------------------------------------------------------------
 * cdata.i
 *
 * SWIG library file containing macros for manipulating raw C data as strings.
 * ----------------------------------------------------------------------------- */

%{
typedef struct SWIGCDATA {
    char *data;
    intgo len;
} SWIGCDATA;
%}

%fragment("cdata", "header") %{
struct swigcdata {
  intgo size;
  void *data;
};
%}

%typemap(gotype) SWIGCDATA "[]byte"

%typemap(imtype) SWIGCDATA "uint64"

%typemap(out, fragment="cdata") SWIGCDATA(struct swigcdata *swig_out) %{
  swig_out = (struct swigcdata *)malloc(sizeof(*swig_out));
  if (swig_out) {
    swig_out->size = $1.len;
    swig_out->data = malloc(swig_out->size);
    if (swig_out->data) {
      memcpy(swig_out->data, $1.data, swig_out->size);
    }
  }
  $result = *(long long *)(void **)&swig_out;
%}

%typemap(goout) SWIGCDATA %{
  {
    type swigcdata struct { size int; data uintptr }
    p := (*swigcdata)(unsafe.Pointer(uintptr($1)))
    if p == nil || p.data == 0 {
      $result = nil
    } else {
      b := make([]byte, p.size)
      a := (*[0x7fffffff]byte)(unsafe.Pointer(p.data))[:p.size]
      copy(b, a)
      Swig_free(p.data)
      Swig_free(uintptr(unsafe.Pointer(p)))
      $result = b
    }
  }
%}

/* -----------------------------------------------------------------------------
 * %cdata(TYPE [, NAME]) 
 *
 * Convert raw C data to a binary string.
 * ----------------------------------------------------------------------------- */

%define %cdata(TYPE,NAME...)

%insert("header") {
#if #NAME == ""
static SWIGCDATA cdata_##TYPE(TYPE *ptr, int nelements) {
#else
static SWIGCDATA cdata_##NAME(TYPE *ptr, int nelements) {
#endif
   SWIGCDATA d;
   d.data = (char *) ptr;
#if #TYPE != "void"
   d.len  = nelements*sizeof(TYPE);
#else
   d.len  = nelements;
#endif
   return d;
}
}

%typemap(default) int nelements "$1 = 1;"

#if #NAME == ""
SWIGCDATA cdata_##TYPE(TYPE *ptr, int nelements);
#else
SWIGCDATA cdata_##NAME(TYPE *ptr, int nelements);
#endif
%enddef

%typemap(default) int nelements;

%rename(cdata) ::cdata_void(void *ptr, int nelements);

%cdata(void);

/* Memory move function. Due to multi-argument typemaps this appears
   to be wrapped as
   void memmove(void *data, const char *s); */
void memmove(void *data, char *indata, int inlen);
