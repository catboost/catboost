/* -----------------------------------------------------------------------------
 * perlrun.swg
 *
 * This file contains the runtime support for Perl modules
 * and includes code for managing global variables and pointer
 * type checking.
 * ----------------------------------------------------------------------------- */

#ifdef PERL_OBJECT
#define SWIG_PERL_OBJECT_DECL CPerlObj *SWIGUNUSEDPARM(pPerl),
#define SWIG_PERL_OBJECT_CALL pPerl,
#else
#define SWIG_PERL_OBJECT_DECL
#define SWIG_PERL_OBJECT_CALL
#endif

/* Common SWIG API */

/* for raw pointers */
#define SWIG_ConvertPtr(obj, pp, type, flags)           SWIG_Perl_ConvertPtr(SWIG_PERL_OBJECT_CALL obj, pp, type, flags)
#define SWIG_ConvertPtrAndOwn(obj, pp, type, flags,own) SWIG_Perl_ConvertPtrAndOwn(SWIG_PERL_OBJECT_CALL obj, pp, type, flags, own)
#define SWIG_NewPointerObj(p, type, flags)              SWIG_Perl_NewPointerObj(SWIG_PERL_OBJECT_CALL p, type, flags)
#define SWIG_AcquirePtr(ptr, src)                       SWIG_Perl_AcquirePtr(ptr, src)
#define swig_owntype                                    int

/* for raw packed data */
#define SWIG_ConvertPacked(obj, p, s, type)             SWIG_Perl_ConvertPacked(SWIG_PERL_OBJECT_CALL obj, p, s, type)
#define SWIG_NewPackedObj(p, s, type)	                SWIG_Perl_NewPackedObj(SWIG_PERL_OBJECT_CALL p, s, type)

/* for class or struct pointers */
#define SWIG_ConvertInstance(obj, pptr, type, flags)    SWIG_ConvertPtr(obj, pptr, type, flags)
#define SWIG_NewInstanceObj(ptr, type, flags)           SWIG_NewPointerObj(ptr, type, flags)

/* for C or C++ function pointers */
#define SWIG_ConvertFunctionPtr(obj, pptr, type)        SWIG_ConvertPtr(obj, pptr, type, 0)
#define SWIG_NewFunctionPtrObj(ptr, type)               SWIG_NewPointerObj(ptr, type, 0)

/* for C++ member pointers, ie, member methods */
#define SWIG_ConvertMember(obj, ptr, sz, ty)            SWIG_ConvertPacked(obj, ptr, sz, ty)
#define SWIG_NewMemberObj(ptr, sz, type)                SWIG_NewPackedObj(ptr, sz, type)


/* Runtime API */

#define SWIG_GetModule(clientdata)                      SWIG_Perl_GetModule(clientdata)
#define SWIG_SetModule(clientdata, pointer)             SWIG_Perl_SetModule(pointer)


/* Error manipulation */

#define SWIG_ErrorType(code)                            SWIG_Perl_ErrorType(code)               
#define SWIG_Error(code, msg)            		sv_setpvf(get_sv("@", GV_ADD), "%s %s", SWIG_ErrorType(code), msg)
#define SWIG_fail                        		goto fail						    

/* Perl-specific SWIG API */

#define SWIG_MakePtr(sv, ptr, type, flags)              SWIG_Perl_MakePtr(SWIG_PERL_OBJECT_CALL sv, ptr, type, flags)
#define SWIG_MakePackedObj(sv, p, s, type)	        SWIG_Perl_MakePackedObj(SWIG_PERL_OBJECT_CALL sv, p, s, type)
#define SWIG_SetError(str)                              SWIG_Error(SWIG_RuntimeError, str)


#define SWIG_PERL_DECL_ARGS_1(arg1)                     (SWIG_PERL_OBJECT_DECL arg1)
#define SWIG_PERL_CALL_ARGS_1(arg1)                     (SWIG_PERL_OBJECT_CALL arg1)
#define SWIG_PERL_DECL_ARGS_2(arg1, arg2)               (SWIG_PERL_OBJECT_DECL arg1, arg2)
#define SWIG_PERL_CALL_ARGS_2(arg1, arg2)               (SWIG_PERL_OBJECT_CALL arg1, arg2)

/* -----------------------------------------------------------------------------
 * pointers/data manipulation
 * ----------------------------------------------------------------------------- */

/* For backward compatibility only */
#define SWIG_POINTER_EXCEPTION  0

#ifdef __cplusplus
extern "C" {
#endif

#define SWIG_OWNER   SWIG_POINTER_OWN
#define SWIG_SHADOW  SWIG_OWNER << 1

#define SWIG_MAYBE_PERL_OBJECT SWIG_PERL_OBJECT_DECL

/* SWIG Perl macros */

/* Macro to declare an XS function */
#ifndef XSPROTO
#   define XSPROTO(name) void name(pTHX_ CV* cv)
#endif

/* Macro to call an XS function */
#ifdef PERL_OBJECT 
#  define SWIG_CALLXS(_name) _name(cv,pPerl) 
#else 
#  ifndef MULTIPLICITY 
#    define SWIG_CALLXS(_name) _name(cv) 
#  else 
#    define SWIG_CALLXS(_name) _name(PERL_GET_THX, cv) 
#  endif 
#endif 

#ifdef PERL_OBJECT
#define MAGIC_PPERL  CPerlObj *pPerl = (CPerlObj *) this;

#ifdef __cplusplus
extern "C" {
#endif
typedef int (CPerlObj::*SwigMagicFunc)(SV *, MAGIC *);
#ifdef __cplusplus
}
#endif

#define SWIG_MAGIC(a,b) (SV *a, MAGIC *b)
#define SWIGCLASS_STATIC

#else /* PERL_OBJECT */

#define MAGIC_PPERL
#define SWIGCLASS_STATIC static SWIGUNUSED

#ifndef MULTIPLICITY
#define SWIG_MAGIC(a,b) (SV *a, MAGIC *b)

#ifdef __cplusplus
extern "C" {
#endif
typedef int (*SwigMagicFunc)(SV *, MAGIC *);
#ifdef __cplusplus
}
#endif

#else /* MULTIPLICITY */

#define SWIG_MAGIC(a,b) (struct interpreter *interp, SV *a, MAGIC *b)

#ifdef __cplusplus
extern "C" {
#endif
typedef int (*SwigMagicFunc)(struct interpreter *, SV *, MAGIC *);
#ifdef __cplusplus
}
#endif

#endif /* MULTIPLICITY */
#endif /* PERL_OBJECT */

#  ifdef PERL_OBJECT
#    define SWIG_croak_null() SWIG_Perl_croak_null(pPerl)
static void SWIGUNUSED SWIG_Perl_croak_null(CPerlObj *pPerl)
#  else
static void SWIGUNUSED SWIG_croak_null()
#  endif
{
  SV *err = get_sv("@", GV_ADD);
#  if (PERL_VERSION < 6)
  croak("%_", err);
#  else
  if (sv_isobject(err))
    croak(0);
  else
    croak("%s", SvPV_nolen(err));
#  endif
}


/* 
   Define how strict is the cast between strings and integers/doubles
   when overloading between these types occurs.
   
   The default is making it as strict as possible by using SWIG_AddCast
   when needed.
   
   You can use -DSWIG_PERL_NO_STRICT_STR2NUM at compilation time to
   disable the SWIG_AddCast, making the casting between string and
   numbers less strict.

   In the end, we try to solve the overloading between strings and
   numerical types in the more natural way, but if you can avoid it,
   well, avoid it using %rename, for example.
*/
#ifndef SWIG_PERL_NO_STRICT_STR2NUM
# ifndef SWIG_PERL_STRICT_STR2NUM
#  define SWIG_PERL_STRICT_STR2NUM
# endif
#endif
#ifdef SWIG_PERL_STRICT_STR2NUM
/* string takes precedence */
#define SWIG_Str2NumCast(x) SWIG_AddCast(x)  
#else
/* number takes precedence */
#define SWIG_Str2NumCast(x) x
#endif



#include <stdlib.h>

SWIGRUNTIME const char *
SWIG_Perl_TypeProxyName(const swig_type_info *type) {
  if (!type) return NULL;
  if (type->clientdata != NULL) {
    return (const char*) type->clientdata;
  } 
  else {
    return type->name;
  }
}

/* Identical to SWIG_TypeCheck, except for strcmp comparison */
SWIGRUNTIME swig_cast_info *
SWIG_TypeProxyCheck(const char *c, swig_type_info *ty) {
  if (ty) {
    swig_cast_info *iter = ty->cast;
    while (iter) {
      if (strcmp(SWIG_Perl_TypeProxyName(iter->type), c) == 0) {
        if (iter == ty->cast)
          return iter;
        /* Move iter to the top of the linked list */
        iter->prev->next = iter->next;
        if (iter->next)
          iter->next->prev = iter->prev;
        iter->next = ty->cast;
        iter->prev = 0;
        if (ty->cast) ty->cast->prev = iter;
        ty->cast = iter;
        return iter;
      }
      iter = iter->next;
    }
  }
  return 0;
}

/* Acquire a pointer value */

SWIGRUNTIME int
SWIG_Perl_AcquirePtr(SWIG_MAYBE_PERL_OBJECT SV *sv, int own) {
  /* TODO */
  return 0;
}

/* Function for getting a pointer value */

SWIGRUNTIME int
SWIG_Perl_ConvertPtrAndOwn(SWIG_MAYBE_PERL_OBJECT SV *sv, void **ptr, swig_type_info *_t, int flags, int *own) {
  swig_cast_info *tc;
  void *voidptr = (void *)0;
  SV *tsv = 0;

  if (own)
    *own = 0;

  /* If magical, apply more magic */
  if (SvGMAGICAL(sv))
    mg_get(sv);

  /* Check to see if this is an object */
  if (sv_isobject(sv)) {
    IV tmp = 0;
    tsv = (SV*) SvRV(sv);
    if ((SvTYPE(tsv) == SVt_PVHV)) {
      MAGIC *mg;
      if (SvMAGICAL(tsv)) {
        mg = mg_find(tsv,'P');
        if (mg) {
          sv = mg->mg_obj;
          if (sv_isobject(sv)) {
	    tsv = (SV*)SvRV(sv);
            tmp = SvIV(tsv);
          }
        }
      } else {
        return SWIG_ERROR;
      }
    } else {
      tmp = SvIV(tsv);
    }
    voidptr = INT2PTR(void *,tmp);
  } else if (! SvOK(sv)) {            /* Check for undef */
    *(ptr) = (void *) 0;
    return (flags & SWIG_POINTER_NO_NULL) ? SWIG_NullReferenceError : SWIG_OK;
  } else if (SvTYPE(sv) == SVt_RV) {  /* Check for NULL pointer */
    if (!SvROK(sv)) {
      /* In Perl 5.12 and later, SVt_RV == SVt_IV, so sv could be a valid integer value.  */
      if (SvIOK(sv)) {
        return SWIG_ERROR;
      } else {
        /* NULL pointer (reference to undef). */
        *(ptr) = (void *) 0;
        return SWIG_OK;
      }
    } else {
      return SWIG_ERROR;
    }
  } else {                            /* Don't know what it is */
    return SWIG_ERROR;
  }
  if (_t) {
    /* Now see if the types match */
    char *_c = HvNAME(SvSTASH(SvRV(sv)));
    tc = SWIG_TypeProxyCheck(_c,_t);
#ifdef SWIG_DIRECTORS
    if (!tc && !sv_derived_from(sv,SWIG_Perl_TypeProxyName(_t))) {
#else
    if (!tc) {
#endif
      return SWIG_ERROR;
    }
    {
      int newmemory = 0;
      *ptr = SWIG_TypeCast(tc,voidptr,&newmemory);
      if (newmemory == SWIG_CAST_NEW_MEMORY) {
        assert(own); /* badly formed typemap which will lead to a memory leak - it must set and use own to delete *ptr */
        if (own)
          *own = *own | SWIG_CAST_NEW_MEMORY;
      }
    }
  } else {
    *ptr = voidptr;
  }

  /* 
   *  DISOWN implementation: we need a perl guru to check this one.
   */
  if (tsv && (flags & SWIG_POINTER_DISOWN)) {
    /* 
     *  almost copy paste code from below SWIG_POINTER_OWN setting
     */
    SV *obj = sv;
    HV *stash = SvSTASH(SvRV(obj));
    GV *gv = *(GV**)hv_fetch(stash, "OWNER", 5, TRUE);
    if (isGV(gv)) {
      HV *hv = GvHVn(gv);
      /*
       * To set ownership (see below), a newSViv(1) entry is added. 
       * Hence, to remove ownership, we delete the entry.
       */
      if (hv_exists_ent(hv, obj, 0)) {
	hv_delete_ent(hv, obj, 0, 0);
      }
    }
  }
  return SWIG_OK;
}

SWIGRUNTIME int
SWIG_Perl_ConvertPtr(SWIG_MAYBE_PERL_OBJECT SV *sv, void **ptr, swig_type_info *_t, int flags) {
  return SWIG_Perl_ConvertPtrAndOwn(sv, ptr, _t, flags, 0);
}

SWIGRUNTIME void
SWIG_Perl_MakePtr(SWIG_MAYBE_PERL_OBJECT SV *sv, void *ptr, swig_type_info *t, int flags) {
  if (ptr && (flags & (SWIG_SHADOW | SWIG_POINTER_OWN))) {
    SV *self;
    SV *obj=newSV(0);
    HV *hash=newHV();
    HV *stash;
    sv_setref_pv(obj, SWIG_Perl_TypeProxyName(t), ptr);
    stash=SvSTASH(SvRV(obj));
    if (flags & SWIG_POINTER_OWN) {
      HV *hv;
      GV *gv = *(GV**)hv_fetch(stash, "OWNER", 5, TRUE);
      if (!isGV(gv))
        gv_init(gv, stash, "OWNER", 5, FALSE);
      hv=GvHVn(gv);
      hv_store_ent(hv, obj, newSViv(1), 0);
    }
    sv_magic((SV *)hash, (SV *)obj, 'P', Nullch, 0);
    SvREFCNT_dec(obj);
    self=newRV_noinc((SV *)hash);
    sv_setsv(sv, self);
    SvREFCNT_dec((SV *)self);
    sv_bless(sv, stash);
  }
  else {
    sv_setref_pv(sv, SWIG_Perl_TypeProxyName(t), ptr);
  }
}

SWIGRUNTIMEINLINE SV *
SWIG_Perl_NewPointerObj(SWIG_MAYBE_PERL_OBJECT void *ptr, swig_type_info *t, int flags) {
  SV *result = sv_newmortal();
  SWIG_MakePtr(result, ptr, t, flags);
  return result;
}

SWIGRUNTIME void
SWIG_Perl_MakePackedObj(SWIG_MAYBE_PERL_OBJECT SV *sv, void *ptr, int sz, swig_type_info *type) {
  char result[1024];
  char *r = result;
  if ((2*sz + 1 + strlen(SWIG_Perl_TypeProxyName(type))) > 1000) return;
  *(r++) = '_';
  r = SWIG_PackData(r,ptr,sz);
  strcpy(r,SWIG_Perl_TypeProxyName(type));
  sv_setpv(sv, result);
}

SWIGRUNTIME SV *
SWIG_Perl_NewPackedObj(SWIG_MAYBE_PERL_OBJECT void *ptr, int sz, swig_type_info *type) {
  SV *result = sv_newmortal();
  SWIG_Perl_MakePackedObj(result, ptr, sz, type);
  return result;
}

/* Convert a packed pointer value */
SWIGRUNTIME int
SWIG_Perl_ConvertPacked(SWIG_MAYBE_PERL_OBJECT SV *obj, void *ptr, int sz, swig_type_info *ty) {
  swig_cast_info *tc;
  const char  *c = 0;

  if ((!obj) || (!SvOK(obj))) return SWIG_ERROR;
  c = SvPV_nolen(obj);
  /* Pointer values must start with leading underscore */
  if (*c != '_') return SWIG_ERROR;
  c++;
  c = SWIG_UnpackData(c,ptr,sz);
  if (ty) {
    tc = SWIG_TypeCheck(c,ty);
    if (!tc) return SWIG_ERROR;
  }
  return SWIG_OK;
}


/* Macros for low-level exception handling */
#define SWIG_croak(x)    { SWIG_Error(SWIG_RuntimeError, x); SWIG_fail; }


typedef XSPROTO(SwigPerlWrapper);
typedef SwigPerlWrapper *SwigPerlWrapperPtr;

/* Structure for command table */
typedef struct {
  const char         *name;
  SwigPerlWrapperPtr  wrapper;
} swig_command_info;

/* Information for constant table */

#define SWIG_INT     1
#define SWIG_FLOAT   2
#define SWIG_STRING  3
#define SWIG_POINTER 4
#define SWIG_BINARY  5

/* Constant information structure */
typedef struct swig_constant_info {
    int              type;
    const char      *name;
    long             lvalue;
    double           dvalue;
    void            *pvalue;
    swig_type_info **ptype;
} swig_constant_info;


/* Structure for variable table */
typedef struct {
  const char   *name;
  SwigMagicFunc   set;
  SwigMagicFunc   get;
  swig_type_info  **type;
} swig_variable_info;

/* Magic variable code */
#ifndef PERL_OBJECT
# ifdef __cplusplus
#  define swig_create_magic(s,a,b,c) _swig_create_magic(s,const_cast<char*>(a),b,c)
# else
#  define swig_create_magic(s,a,b,c) _swig_create_magic(s,(char*)(a),b,c)
# endif
# ifndef MULTIPLICITY
SWIGRUNTIME void _swig_create_magic(SV *sv, char *name, int (*set)(SV *, MAGIC *), int (*get)(SV *,MAGIC *)) 
# else
SWIGRUNTIME void _swig_create_magic(SV *sv, char *name, int (*set)(struct interpreter*, SV *, MAGIC *), int (*get)(struct interpreter*, SV *,MAGIC *)) 
# endif
#else
#  define swig_create_magic(s,a,b,c) _swig_create_magic(pPerl,s,a,b,c)
SWIGRUNTIME void _swig_create_magic(CPerlObj *pPerl, SV *sv, const char *name, int (CPerlObj::*set)(SV *, MAGIC *), int (CPerlObj::*get)(SV *, MAGIC *)) 
#endif
{
  MAGIC *mg;
  sv_magic(sv,sv,'U',name,strlen(name));
  mg = mg_find(sv,'U');
  mg->mg_virtual = (MGVTBL *) malloc(sizeof(MGVTBL));
  mg->mg_virtual->svt_get = (SwigMagicFunc) get;
  mg->mg_virtual->svt_set = (SwigMagicFunc) set;
  mg->mg_virtual->svt_len = 0;
  mg->mg_virtual->svt_clear = 0;
  mg->mg_virtual->svt_free = 0;
}


SWIGRUNTIME swig_module_info *
SWIG_Perl_GetModule(void *SWIGUNUSEDPARM(clientdata)) {
  static void *type_pointer = (void *)0;
  SV *pointer;

  /* first check if pointer already created */
  if (!type_pointer) {
    pointer = get_sv("swig_runtime_data::type_pointer" SWIG_RUNTIME_VERSION SWIG_TYPE_TABLE_NAME, FALSE | GV_ADDMULTI);
    if (pointer && SvOK(pointer)) {
      type_pointer = INT2PTR(swig_type_info **, SvIV(pointer));
    }
  }

  return (swig_module_info *) type_pointer;
}

SWIGRUNTIME void
SWIG_Perl_SetModule(swig_module_info *module) {
  SV *pointer;

  /* create a new pointer */
  pointer = get_sv("swig_runtime_data::type_pointer" SWIG_RUNTIME_VERSION SWIG_TYPE_TABLE_NAME, TRUE | GV_ADDMULTI);
  sv_setiv(pointer, PTR2IV(module));
}

#ifdef __cplusplus
}
#endif
