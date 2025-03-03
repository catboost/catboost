/* Remove global namespace pollution */
#if !defined(SWIG_NO_R_NO_REMAP)
# define R_NO_REMAP
#endif
#if !defined(SWIG_NO_STRICT_R_HEADERS)
# define STRICT_R_HEADERS
#endif

#include <Rdefines.h>
#include <Rversion.h>

#ifdef __cplusplus
#include <exception>
extern "C" {
#endif

/* for raw pointer */
#define SWIG_ConvertPtr(obj, pptr, type, flags)         SWIG_R_ConvertPtr(obj, pptr, type, flags)
#define SWIG_ConvertPtrAndOwn(obj,pptr,type,flags,own)  SWIG_R_ConvertPtr(obj, pptr, type, flags)
#define SWIG_NewPointerObj(ptr, type, flags)            SWIG_R_NewPointerObj(ptr, type, flags)

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdarg.h>

#if R_VERSION >= R_Version(2,6,0)
#define VMAXTYPE void *
#else
#define VMAXTYPE char *
#endif

/* Last error */
static int SWIG_lasterror_code = 0;
static char SWIG_lasterror_msg[1024];
SWIGRUNTIME void SWIG_Error(int code, const char *format, ...) {
  va_list arg;
  SWIG_lasterror_code = code;
  va_start(arg, format);
  vsnprintf(SWIG_lasterror_msg, sizeof(SWIG_lasterror_msg), format, arg);
  va_end(arg);
}

SWIGRUNTIME const char *SWIG_ErrorType(int code) {
  switch (code) {
  case SWIG_MemoryError:
    return "SWIG:MemoryError";
  case SWIG_IOError:
    return "SWIG:IOError";
  case SWIG_RuntimeError:
    return "SWIG:RuntimeError";
  case SWIG_IndexError:
    return "SWIG:IndexError";
  case SWIG_TypeError:
    return "SWIG:TypeError";
  case SWIG_DivisionByZero:
    return "SWIG:DivisionByZero";
  case SWIG_OverflowError:
    return "SWIG:OverflowError";
  case SWIG_SyntaxError:
    return "SWIG:SyntaxError";
  case SWIG_ValueError:
    return "SWIG:ValueError";
  case SWIG_SystemError:
    return "SWIG:SystemError";
  case SWIG_AttributeError:
    return "SWIG:AttributeError";
  }
  return "SWIG:UnknownError";
}

#define SWIG_fail goto fail

/*
  This is mainly a way to avoid having lots of local variables that may 
  conflict with those in the routine.

   Change name to R_SWIG_Callb....
*/
typedef struct RCallbackFunctionData {

  SEXP fun;
  SEXP userData;


  SEXP expr;
  SEXP retValue;
  int errorOccurred;

  SEXP el;  /* Temporary pointer used in the construction of the expression to call the R function. */

  struct RCallbackFunctionData *previous;   /* Stack */

} RCallbackFunctionData;

static RCallbackFunctionData  *callbackFunctionDataStack;


SWIGRUNTIME SEXP
R_SWIG_debug_getCallbackFunctionData()
{
  int n, i;
  SEXP ans;
  RCallbackFunctionData  *p = callbackFunctionDataStack;

  n = 0;
  while(p) { 
    n++;
    p = p->previous;
  }

  Rf_protect(ans = Rf_allocVector(VECSXP, n));
  for(p = callbackFunctionDataStack, i = 0; i < n; p = p->previous, i++) 
      SET_VECTOR_ELT(ans, i, p->fun);

  Rf_unprotect(1);

  return(ans);
}



SWIGRUNTIME RCallbackFunctionData *
R_SWIG_pushCallbackFunctionData(SEXP fun, SEXP userData)
{
   RCallbackFunctionData *el;
   el = (RCallbackFunctionData *) calloc(1, sizeof(RCallbackFunctionData));
   el->fun = fun;
   el->userData = userData;
   el->previous = callbackFunctionDataStack;

   callbackFunctionDataStack = el;

   return(el);
}


SWIGRUNTIME SEXP
R_SWIG_R_pushCallbackFunctionData(SEXP fun, SEXP userData)
{
    R_SWIG_pushCallbackFunctionData(fun, userData);
    return R_NilValue;
}

SWIGRUNTIME RCallbackFunctionData *
R_SWIG_getCallbackFunctionData()
{
  if(!callbackFunctionDataStack) {
    Rf_error("Supposedly impossible error occurred in the SWIG callback mechanism."
            "  No callback function data set.");
  }
  
  return callbackFunctionDataStack;
}

SWIGRUNTIME void
R_SWIG_popCallbackFunctionData(int doFree)
{
  RCallbackFunctionData  *el = NULL;
  if(!callbackFunctionDataStack)
    return ; /* Error !!! */

  el = callbackFunctionDataStack ;
  callbackFunctionDataStack = callbackFunctionDataStack->previous;

  if(doFree)
     free(el);
}


/*
  Interface to S function
      is(obj, type)
  which is to be used to determine if an 
  external pointer inherits from the right class.

  Ideally, we would like to be able to do this without an explicit call to the is() function.
  When the S4 class system uses its own SEXP types, then we will hopefully be able to do this
  in the C code.

  Should we make the expression static and preserve it to avoid the overhead of 
  allocating each time.
*/
SWIGRUNTIME int
R_SWIG_checkInherits(SEXP obj, SEXP tag, const char *type)
{
  SEXP e, val;
  int check_err = 0;

  Rf_protect(e = Rf_allocVector(LANGSXP, 3));
  SETCAR(e, Rf_install("extends"));

  SETCAR(CDR(e), Rf_mkString(CHAR(PRINTNAME(tag))));
  SETCAR(CDR(CDR(e)), Rf_mkString(type));

  val = R_tryEval(e, R_GlobalEnv, &check_err);
  Rf_unprotect(1);
  if(check_err) 
    return(0);


  return(LOGICAL(val)[0]);
}


SWIGRUNTIME void *
R_SWIG_resolveExternalRef(SEXP arg, const char * const type, const char * const argName, Rboolean nullOk)
{
  void *ptr;
  SEXP orig = arg;

  if(TYPEOF(arg) != EXTPTRSXP) 
    arg = GET_SLOT(arg, Rf_mkString("ref"));

  
  if(TYPEOF(arg) != EXTPTRSXP) {
    Rf_error("argument %s must be an external pointer (from an ExternalReference)", argName);
  }


  ptr = R_ExternalPtrAddr(arg);

  if(ptr == NULL && nullOk == (Rboolean) FALSE) {
    Rf_error("the external pointer (of type %s) for argument %s has value NULL", argName, type);
  }

  if(type[0] && R_ExternalPtrTag(arg) != Rf_install(type) && strcmp(type, "voidRef")
      && !R_SWIG_checkInherits(orig,  R_ExternalPtrTag(arg), type)) {
    Rf_error("the external pointer for argument %s has tag %s, not the expected value %s",
             argName, CHAR(PRINTNAME(R_ExternalPtrTag(arg))), type);
  }


  return(ptr);
}

SWIGRUNTIME void
R_SWIG_ReferenceFinalizer(SEXP el)
{
  void *ptr = R_SWIG_resolveExternalRef(el, "", "<finalizer>",  (Rboolean) 1);
  fprintf(stderr, "In R_SWIG_ReferenceFinalizer for %p\n", ptr);
  Rf_PrintValue(el);

  if(ptr) {
     if(TYPEOF(el) != EXTPTRSXP)
        el = GET_SLOT(el, Rf_mkString("ref"));

     if(TYPEOF(el) == EXTPTRSXP)
        R_ClearExternalPtr(el);

     free(ptr);
  }

  return;
}

SWIGRUNTIME SEXP
SWIG_MakePtr(void *ptr, const char *typeName, int flags)
{
  SEXP external, r_obj;

  Rf_protect(external = R_MakeExternalPtr(ptr, Rf_install(typeName), R_NilValue));
  Rf_protect(r_obj = NEW_OBJECT(MAKE_CLASS((char *) typeName)));

  if (flags & SWIG_POINTER_OWN)
    R_RegisterCFinalizer(external, R_SWIG_ReferenceFinalizer);

  r_obj = SET_SLOT(r_obj, Rf_mkString((char *) "ref"), external);
  SET_S4_OBJECT(r_obj);
  Rf_unprotect(2);

  return(r_obj);
}


SWIGRUNTIME SEXP
R_SWIG_create_SWIG_R_Array(const char *typeName, SEXP ref, int len)
{
   SEXP arr;

/*XXX remove the char * cast when we can. MAKE_CLASS should be declared appropriately. */
   Rf_protect(arr = NEW_OBJECT(MAKE_CLASS((char *) typeName)));
   Rf_protect(arr = R_do_slot_assign(arr, Rf_mkString("ref"), ref));
   Rf_protect(arr = R_do_slot_assign(arr, Rf_mkString("dims"), Rf_ScalarInteger(len)));

   Rf_unprotect(3); 			   
   SET_S4_OBJECT(arr);	
   return arr;
}

#define ADD_OUTPUT_ARG(result, pos, value, name)  r_ans = AddOutputArgToReturn(pos, value, name, OutputValues);

SWIGRUNTIME SEXP
AddOutputArgToReturn(int pos, SEXP value, const char *name, SEXP output)
{
  SET_VECTOR_ELT(output, pos, value);

  return(output);
}

/* Create a new pointer object */
SWIGRUNTIMEINLINE SEXP
SWIG_R_NewPointerObj(void *ptr, swig_type_info *type, int flags) {
  SEXP rptr;
  if (!ptr) {
     return R_NilValue;
  }
  rptr = R_MakeExternalPtr(ptr, 
  R_MakeExternalPtr(type, R_NilValue, R_NilValue), R_NilValue); 
  SET_S4_OBJECT(rptr);
  return rptr;
}


/* Convert a pointer value */
SWIGRUNTIMEINLINE int
SWIG_R_ConvertPtr(SEXP obj, void **ptr, swig_type_info *ty, int flags) {
  void *vptr;
  if (!obj) return SWIG_ERROR;
  if (obj == R_NilValue) {
    if (ptr) *ptr = NULL;
    return (flags & SWIG_POINTER_NO_NULL) ? SWIG_NullReferenceError : SWIG_OK;
  }

  vptr = R_ExternalPtrAddr(obj);
  if (ty) {
    swig_type_info *to = (swig_type_info*) 
      R_ExternalPtrAddr(R_ExternalPtrTag(obj));
    if (to == ty) {
      if (ptr) *ptr = vptr;
    } else {
      swig_cast_info *tc = SWIG_TypeCheck(to->name,ty);
      int newmemory = 0;
      if (ptr) *ptr = SWIG_TypeCast(tc,vptr,&newmemory);
      assert(!newmemory); /* newmemory handling not yet implemented */
    }
  } else {
      if (ptr) *ptr = vptr;
 }
  return SWIG_OK;
}

SWIGRUNTIME swig_module_info *
SWIG_GetModule(void *SWIGUNUSEDPARM(clientdata)) {
  static void *type_pointer = (void *)0;
  return (swig_module_info *) type_pointer;
}

SWIGRUNTIME void
SWIG_SetModule(void *v, swig_module_info *swig_module) {
}

typedef struct {
  void *pack;
  swig_type_info *ty;
  size_t size;
} RSwigPacked;

/* Create a new packed object */

SWIGRUNTIMEINLINE SEXP RSwigPacked_New(void *ptr, size_t sz,
		  swig_type_info *ty) {
  SEXP rptr;
  RSwigPacked *sobj = 
  (RSwigPacked*) malloc(sizeof(RSwigPacked));
  if (sobj) {
    void *pack = malloc(sz);
    if (pack) {
      memcpy(pack, ptr, sz);
      sobj->pack = pack;
      sobj->ty   = ty;
      sobj->size = sz;
    } else {
      sobj = 0;
    }
  }
  rptr = R_MakeExternalPtr(sobj, R_NilValue, R_NilValue); 
  return rptr;
}

SWIGRUNTIME swig_type_info *
RSwigPacked_UnpackData(SEXP obj, void *ptr, size_t size)
{
    RSwigPacked *sobj = 
        (RSwigPacked *)R_ExternalPtrAddr(obj);
    if (sobj->size != size) return 0;
    memcpy(ptr, sobj->pack, size);
    return sobj->ty;
}

SWIGRUNTIMEINLINE SEXP
SWIG_R_NewPackedObj(void *ptr, size_t sz, swig_type_info *type) {
  return ptr ? RSwigPacked_New((void *) ptr, sz, type) : R_NilValue;
}

/* Convert a packed pointer value */

SWIGRUNTIME int
SWIG_R_ConvertPacked(SEXP obj, void *ptr, size_t sz, swig_type_info *ty) {
  swig_type_info *to = RSwigPacked_UnpackData(obj, ptr, sz);
  if (!to) return SWIG_ERROR;
  if (ty) {
    if (to != ty) {
      /* check type cast? */
      swig_cast_info *tc = SWIG_TypeCheck(to->name,ty);
      if (!tc) return SWIG_ERROR;
    }
  }
  return SWIG_OK;
}  

#ifdef __cplusplus
#define SWIG_exception_noreturn(code, msg) do { throw std::runtime_error(msg); } while(0)
#else
#define SWIG_exception_noreturn(code, msg) do { return result; } while(0)
#endif

#ifdef __cplusplus
}
#endif
