/* -----------------------------------------------------------------------------
 * guile_scm_run.swg
 * ----------------------------------------------------------------------------- */

#include <libguile.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif


/* In the code below, use guile 2.0 compatible functions where possible.
   Functions that don't exist in older versions will be mapped to
   a deprecated equivalent for those versions only */
#if defined (SCM_MAJOR_VERSION) && (SCM_MAJOR_VERSION < 2)

static SCM
scm_module_variable (SCM module, SCM sym)
{
  return scm_sym2var (sym, scm_module_lookup_closure (module), SCM_BOOL_F);
}

#define scm_to_utf8_string scm_to_locale_string
#define scm_from_utf8_string scm_from_locale_string
#endif

#if SCM_MAJOR_VERSION >= 2
/* scm_c_define_gsubr takes a different parameter type depending on the guile version */

typedef scm_t_subr swig_guile_proc;
#else
typedef SCM (*swig_guile_proc)();
#endif
typedef SCM (*guile_destructor)(SCM);

typedef struct swig_guile_clientdata {
  guile_destructor destroy;
  SCM goops_class;
} swig_guile_clientdata;

#define SWIG_scm2str(s) \
  SWIG_Guile_scm2newstr(s, NULL)
#define SWIG_str02scm(str) \
  str ? scm_from_utf8_string(str) : SCM_BOOL_F 
# define SWIG_malloc(size) \
  scm_malloc(size)
# define SWIG_free(mem) \
  free(mem)
#define SWIG_ConvertPtr(s, result, type, flags) \
  SWIG_Guile_ConvertPtr(s, result, type, flags)
#define SWIG_MustGetPtr(s, type, argnum, flags) \
  SWIG_Guile_MustGetPtr(s, type, argnum, flags, FUNC_NAME)
#define SWIG_NewPointerObj(ptr, type, owner) \
  SWIG_Guile_NewPointerObj((void*)ptr, type, owner)
#define SWIG_PointerAddress(object) \
  SWIG_Guile_PointerAddress(object)
#define SWIG_PointerType(object) \
  SWIG_Guile_PointerType(object)
#define SWIG_IsPointerOfType(object, type) \
  SWIG_Guile_IsPointerOfType(object, type)
#define SWIG_IsPointer(object) \
  SWIG_Guile_IsPointer(object)
#define SWIG_contract_assert(expr, msg)				\
  if (!(expr))							\
    scm_error(scm_from_locale_symbol("swig-contract-assertion-failed"),	\
	      (char *) FUNC_NAME, (char *) msg,			\
	      SCM_EOL, SCM_BOOL_F); else

/* for C++ member pointers, ie, member methods */
#define SWIG_ConvertMember(obj, ptr, sz, ty) \
  SWIG_Guile_ConvertMember(obj, ptr, sz, ty, FUNC_NAME)
#define SWIG_NewMemberObj(ptr, sz, type) \
  SWIG_Guile_NewMemberObj(ptr, sz, type, FUNC_NAME)
  
/* Runtime API */
static swig_module_info *SWIG_Guile_GetModule(void *SWIGUNUSEDPARM(clientdata));
#define SWIG_GetModule(clientdata) SWIG_Guile_GetModule(clientdata)
#define SWIG_SetModule(clientdata, pointer) SWIG_Guile_SetModule(pointer)
  
SWIGINTERN char *
SWIG_Guile_scm2newstr(SCM str, size_t *len) {
#define FUNC_NAME "SWIG_Guile_scm2newstr"
  char *ret;

  SCM_ASSERT (scm_is_string(str), str, 1, FUNC_NAME);

  ret = scm_to_utf8_string(str);
  if (!ret) return NULL;

  if (len) *len = strlen(ret) - 1;
  return ret;
#undef FUNC_NAME
}

static int swig_initialized = 0;
static scm_t_bits swig_tag = 0;
static scm_t_bits swig_collectable_tag = 0;
static scm_t_bits swig_finalized_tag = 0;
static scm_t_bits swig_destroyed_tag = 0;
static scm_t_bits swig_member_function_tag = 0;
static SCM swig_make_func = SCM_EOL;
static SCM swig_keyword = SCM_EOL;
static SCM swig_symbol = SCM_EOL;

#define SWIG_Guile_GetSmob(x) \
  ( !scm_is_null(x) && SCM_INSTANCEP(x) && scm_is_true(scm_slot_exists_p(x, swig_symbol)) \
      ? scm_slot_ref(x, swig_symbol) : (x) )

SWIGINTERN SCM
SWIG_Guile_NewPointerObj(void *ptr, swig_type_info *type, int owner)
{
  if (ptr == NULL)
    return SCM_EOL;
  else {
    SCM smob;
    swig_guile_clientdata *cdata = (swig_guile_clientdata *) type->clientdata;
    if (owner)
      SCM_NEWSMOB2(smob, swig_collectable_tag, ptr, (void *) type);
    else
      SCM_NEWSMOB2(smob, swig_tag, ptr, (void *) type);

    if (!cdata || SCM_NULLP(cdata->goops_class) || swig_make_func == SCM_EOL ) {
      return smob;
    } else {
      /* the scm_make() C function only handles the creation of gf,
	 methods and classes (no instances) the (make ...) function is
	 later redefined in goops.scm.  So we need to call that
	 Scheme function. */
      return scm_apply(swig_make_func,
		       scm_list_3(cdata->goops_class,
				  swig_keyword,
				  smob),
		       SCM_EOL);
    }
  }
}

SWIGINTERN unsigned long
SWIG_Guile_PointerAddress(SCM object)
{
  SCM smob = SWIG_Guile_GetSmob(object);
  if (SCM_NULLP(smob)) return 0;
  else if (SCM_SMOB_PREDICATE(swig_tag, smob)
	   || SCM_SMOB_PREDICATE(swig_collectable_tag, smob)
	   || SCM_SMOB_PREDICATE(swig_destroyed_tag, smob)) {
    return (unsigned long) (void *) SCM_CELL_WORD_1(smob);
  }
  else scm_wrong_type_arg("SWIG-Guile-PointerAddress", 1, object);
}

SWIGINTERN swig_type_info *
SWIG_Guile_PointerType(SCM object)
{
  SCM smob = SWIG_Guile_GetSmob(object);
  if (SCM_NULLP(smob)) return NULL;
  else if (SCM_SMOB_PREDICATE(swig_tag, smob)
	   || SCM_SMOB_PREDICATE(swig_collectable_tag, smob)
	   || SCM_SMOB_PREDICATE(swig_destroyed_tag, smob)) {
    return (swig_type_info *) SCM_CELL_WORD_2(smob);
  }
  else scm_wrong_type_arg("SWIG-Guile-PointerType", 1, object);
}

SWIGINTERN int
SWIG_Guile_IsValidSmob(SCM smob)
{
  /* We do not accept smobs representing destroyed pointers, but we have to
     allow finalized smobs because Guile >= 2.0.12 sets all smob instances
     to the 'finalized' type before calling their 'free' function. This change
     was introduced to Guile in commit 8dff3af087c6eaa83ae0d72aa8b22aef5c65d65d */
  return SCM_SMOB_PREDICATE(swig_tag, smob)
    || SCM_SMOB_PREDICATE(swig_collectable_tag, smob)
    || SCM_SMOB_PREDICATE(swig_finalized_tag, smob);
}

SWIGINTERN int
SWIG_Guile_ConvertPtr(SCM s, void **result, swig_type_info *type, int flags)
{
  swig_cast_info *cast;
  swig_type_info *from;
  SCM smob = SWIG_Guile_GetSmob(s);

  if (SCM_NULLP(smob)) {
    *result = NULL;
    return (flags & SWIG_POINTER_NO_NULL) ? SWIG_NullReferenceError : SWIG_OK;
#if SCM_MAJOR_VERSION >= 2
  } else if (SCM_POINTER_P(s)) {
    *result = SCM_POINTER_VALUE(s);
    return SWIG_OK;
#endif /* if SCM_MAJOR_VERSION >= 2 */
  } else if (SWIG_Guile_IsValidSmob(smob)) {
    from = (swig_type_info *) SCM_CELL_WORD_2(smob);
    if (!from) return SWIG_ERROR;
    if (type) {
      cast = SWIG_TypeCheckStruct(from, type);
      if (cast) {
        int newmemory = 0;
        *result = SWIG_TypeCast(cast, (void *) SCM_CELL_WORD_1(smob), &newmemory);
        assert(!newmemory); /* newmemory handling not yet implemented */
        return SWIG_OK;
      } else {
        return SWIG_ERROR;
      }
    } else {
      *result = (void *) SCM_CELL_WORD_1(smob);
      return SWIG_OK;
    }
  }
  return SWIG_ERROR;
}

SWIGINTERNINLINE void *
SWIG_Guile_MustGetPtr (SCM s, swig_type_info *type,
		       int argnum, int flags, const char *func_name)
{
  void *result;
  int res = SWIG_Guile_ConvertPtr(s, &result, type, flags);
  if (!SWIG_IsOK(res)) {
    /* type mismatch */
    scm_wrong_type_arg((char *) func_name, argnum, s);
  }
  return result;
}

SWIGINTERNINLINE int
SWIG_Guile_IsPointerOfType (SCM s, swig_type_info *type)
{
  void *result;
  if (SWIG_Guile_ConvertPtr(s, &result, type, 0)) {
    /* type mismatch */
    return 0;
  }
  else return 1;
}

SWIGINTERNINLINE int
SWIG_Guile_IsPointer (SCM s)
{
  /* module might not be initialized yet, so initialize it */
  SWIG_GetModule(0);
  return SWIG_Guile_IsPointerOfType (s, NULL);
}

/* Mark a pointer object non-collectable */
SWIGINTERN void
SWIG_Guile_MarkPointerNoncollectable(SCM s)
{
  SCM smob = SWIG_Guile_GetSmob(s);
  if (!SCM_NULLP(smob)) {
    if (SWIG_Guile_IsValidSmob(smob)) {
      SCM_SET_CELL_TYPE(smob, swig_tag);
    }
    else scm_wrong_type_arg(NULL, 0, s);
  }
}

/* Mark a pointer object destroyed */
SWIGINTERN void
SWIG_Guile_MarkPointerDestroyed(SCM s)
{
  SCM smob = SWIG_Guile_GetSmob(s);
  if (!SCM_NULLP(smob)) {
    if (SWIG_Guile_IsValidSmob(smob)) {
      SCM_SET_CELL_TYPE(smob, swig_destroyed_tag);
    }
    else scm_wrong_type_arg(NULL, 0, s);
  }
}

/* Member functions */

SWIGINTERN SCM
SWIG_Guile_NewMemberObj(void *ptr, size_t sz, swig_type_info *type,
			const char *func_name)
{
  SCM smob;
  void *copy = malloc(sz);
  memcpy(copy, ptr, sz);
  SCM_NEWSMOB2(smob, swig_member_function_tag, copy, (void *) type);
  return smob;
}

SWIGINTERN int
SWIG_Guile_ConvertMember(SCM smob, void *ptr, size_t sz, swig_type_info *type,
			 const char *func_name)
{
  swig_cast_info *cast;
  swig_type_info *from;

  if (SCM_SMOB_PREDICATE(swig_member_function_tag, smob)) {
    from = (swig_type_info *) SCM_CELL_WORD_2(smob);
    if (!from) return SWIG_ERROR;
    if (type) {
      cast = SWIG_TypeCheckStruct(from, type);
      if (!cast) return SWIG_ERROR;
    }
    memcpy(ptr, (void *) SCM_CELL_WORD_1(smob), sz);
    return SWIG_OK;
  }
  return SWIG_ERROR;
}
     

/* Init */

SWIGINTERN int
print_swig_aux (SCM swig_smob, SCM port, scm_print_state *pstate, 
                const char *attribute)
{
  swig_type_info *type;
  
  type = (swig_type_info *) SCM_CELL_WORD_2(swig_smob);
  if (type) {
    scm_puts((char *) "#<", port);
    scm_puts((char *) attribute, port);
    scm_puts((char *) "swig-pointer ", port);
    scm_puts((char *) SWIG_TypePrettyName(type), port);
    scm_puts((char *) " ", port);
    scm_intprint((long) SCM_CELL_WORD_1(swig_smob), 16, port);
    scm_puts((char *) ">", port);
    /* non-zero means success */
    return 1;
  } else {
    return 0;
  }
}

  
SWIGINTERN int
print_swig (SCM swig_smob, SCM port, scm_print_state *pstate)
{
  return print_swig_aux(swig_smob, port, pstate, "");
}

SWIGINTERN int
print_collectable_swig (SCM swig_smob, SCM port, scm_print_state *pstate)
{
  return print_swig_aux(swig_smob, port, pstate, "collectable-");
}

SWIGINTERN int
print_destroyed_swig (SCM swig_smob, SCM port, scm_print_state *pstate)
{
  return print_swig_aux(swig_smob, port, pstate, "destroyed-");
}

SWIGINTERN int
print_member_function_swig (SCM swig_smob, SCM port, scm_print_state *pstate)
{
  swig_type_info *type;
  type = (swig_type_info *) SCM_CELL_WORD_2(swig_smob);
  if (type) {
    scm_puts((char *) "#<", port);
    scm_puts((char *) "swig-member-function-pointer ", port);
    scm_puts((char *) SWIG_TypePrettyName(type), port);
    scm_puts((char *) " >", port);
    /* non-zero means success */
    return 1;
  } else {
    return 0;
  }
}

SWIGINTERN SCM
equalp_swig (SCM A, SCM B)
{
  if (SCM_CELL_WORD_0(A) == SCM_CELL_WORD_0(B) && SCM_CELL_WORD_1(A) == SCM_CELL_WORD_1(B) 
      && SCM_CELL_WORD_2(A) == SCM_CELL_WORD_2(B))
    return SCM_BOOL_T;
  else return SCM_BOOL_F;
}

SWIGINTERN size_t
free_swig(SCM A)
{
  swig_type_info *type = (swig_type_info *) SCM_CELL_WORD_2(A);
  if (type) {
    if (type->clientdata && ((swig_guile_clientdata *)type->clientdata)->destroy)
      ((swig_guile_clientdata *)type->clientdata)->destroy(A);
  } 
  return 0;
}

SWIGINTERN size_t
free_swig_member_function(SCM A)
{
  free((swig_type_info *) SCM_CELL_WORD_1(A));
  return 0;
}

SWIGINTERN int
ensure_smob_tag(SCM swig_module,
		scm_t_bits *tag_variable,
		const char *smob_name,
		const char *scheme_variable_name)
{
  SCM variable = scm_module_variable(swig_module,
                             scm_from_locale_symbol(scheme_variable_name));
  if (scm_is_false(variable)) {
    *tag_variable = scm_make_smob_type((char*)scheme_variable_name, 0);
    scm_c_module_define(swig_module, scheme_variable_name, 
                        scm_from_ulong(*tag_variable));
    return 1;
  }
  else {
    *tag_variable = scm_to_ulong(SCM_VARIABLE_REF(variable));
    return 0;
  }
}

SWIGINTERN SCM
SWIG_Guile_Init ()
{
  static SCM swig_module;
  
  if (swig_initialized) return swig_module;
  swig_initialized = 1;

  swig_module = scm_c_resolve_module("Swig swigrun");
  if (ensure_smob_tag(swig_module, &swig_tag,
		      "swig-pointer", "swig-pointer-tag")) {
    scm_set_smob_print(swig_tag, print_swig);
    scm_set_smob_equalp(swig_tag, equalp_swig);
  }
  if (ensure_smob_tag(swig_module, &swig_collectable_tag,
		      "collectable-swig-pointer", "collectable-swig-pointer-tag")) {
    scm_set_smob_print(swig_collectable_tag, print_collectable_swig);
    scm_set_smob_equalp(swig_collectable_tag, equalp_swig);
    scm_set_smob_free(swig_collectable_tag, free_swig);
    /* For Guile >= 2.0.12. See libguile/smob.c:clear_smobnum */
    swig_finalized_tag = swig_collectable_tag & ~0xff00;
  }
  if (ensure_smob_tag(swig_module, &swig_destroyed_tag,
		      "destroyed-swig-pointer", "destroyed-swig-pointer-tag")) {
    scm_set_smob_print(swig_destroyed_tag, print_destroyed_swig);
    scm_set_smob_equalp(swig_destroyed_tag, equalp_swig);
  }
  if (ensure_smob_tag(swig_module, &swig_member_function_tag,
		      "swig-member-function-pointer", "swig-member-function-pointer-tag")) {
    scm_set_smob_print(swig_member_function_tag, print_member_function_swig);
    scm_set_smob_free(swig_member_function_tag, free_swig_member_function);
  }
  swig_make_func = scm_permanent_object(
  scm_variable_ref(scm_c_module_lookup(scm_c_resolve_module("oop goops"), "make")));
  swig_keyword = scm_permanent_object(scm_from_locale_keyword((char*) "init-smob"));
  swig_symbol = scm_permanent_object(scm_from_locale_symbol("swig-smob"));
#ifdef SWIG_INIT_RUNTIME_MODULE
  SWIG_INIT_RUNTIME_MODULE
#endif

  return swig_module;
}

SWIGINTERN swig_module_info *
SWIG_Guile_GetModule(void *SWIGUNUSEDPARM(clientdata))
{
  SCM module = SWIG_Guile_Init();
  SCM variable = scm_module_variable(module, scm_from_locale_symbol("swig-type-list-address" SWIG_RUNTIME_VERSION SWIG_TYPE_TABLE_NAME));
  if (scm_is_false(variable)) {
    return NULL;
  } else {
    return (swig_module_info *) scm_to_ulong(SCM_VARIABLE_REF(variable));
  }
}

SWIGINTERN void
SWIG_Guile_SetModule(swig_module_info *swig_module)
{
  SCM module = SWIG_Guile_Init();
  scm_module_define(module,
                    scm_from_locale_symbol("swig-type-list-address" SWIG_RUNTIME_VERSION SWIG_TYPE_TABLE_NAME),
                    scm_from_ulong((unsigned long) swig_module));
}

SWIGINTERN int
SWIG_Guile_GetArgs (SCM *dest, SCM rest,
		    int reqargs, int optargs,
		    const char *procname)
{
  int i;
  int num_args_passed = 0;
  for (i = 0; i<reqargs; i++) {
    if (!SCM_CONSP(rest))
      scm_wrong_num_args(scm_from_utf8_string(procname ? (char *) procname : "unknown procedure"));
    *dest++ = SCM_CAR(rest);
    rest = SCM_CDR(rest);
    num_args_passed++;
  }
  for (i = 0; i<optargs && SCM_CONSP(rest); i++) {
    *dest++ = SCM_CAR(rest);
    rest = SCM_CDR(rest);
    num_args_passed++;
  }
  for (; i<optargs; i++)
    *dest++ = SCM_UNDEFINED;
  if (!SCM_NULLP(rest))
      scm_wrong_num_args(scm_from_utf8_string(procname ? (char *) procname : "unknown procedure"));
  return num_args_passed;
}

#ifdef __cplusplus
}
#endif
