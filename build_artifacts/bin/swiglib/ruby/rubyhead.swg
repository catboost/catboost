#include <ruby.h>

/* Ruby 1.9.1 has a "memoisation optimisation" when compiling with GCC which
 * breaks using rb_intern as an lvalue, as SWIG does.  We work around this
 * issue for now by disabling this.
 * https://sourceforge.net/tracker/?func=detail&aid=2859614&group_id=1645&atid=101645
 */
#ifdef rb_intern
# undef rb_intern
#endif

/* Remove global macros defined in Ruby's win32.h */
#ifdef write
# undef write
#endif
#ifdef read
# undef read
#endif
#ifdef bind
# undef bind
#endif
#ifdef close
# undef close
#endif
#ifdef connect
# undef connect
#endif


/* Ruby 1.7 defines NUM2LL(), LL2NUM() and ULL2NUM() macros */
#ifndef NUM2LL
#define NUM2LL(x) NUM2LONG((x))
#endif
#ifndef LL2NUM
#define LL2NUM(x) INT2NUM((long) (x))
#endif
#ifndef ULL2NUM
#define ULL2NUM(x) UINT2NUM((unsigned long) (x))
#endif

/* Ruby 1.7 doesn't (yet) define NUM2ULL() */
#ifndef NUM2ULL
#ifdef HAVE_LONG_LONG
#define NUM2ULL(x) rb_num2ull((x))
#else
#define NUM2ULL(x) NUM2ULONG(x)
#endif
#endif

/* RSTRING_LEN, etc are new in Ruby 1.9, but ->ptr and ->len no longer work */
/* Define these for older versions so we can just write code the new way */
#ifndef RSTRING_LEN
# define RSTRING_LEN(x) RSTRING(x)->len
#endif
#ifndef RSTRING_PTR
# define RSTRING_PTR(x) RSTRING(x)->ptr
#endif
#ifndef RSTRING_END
# define RSTRING_END(x) (RSTRING_PTR(x) + RSTRING_LEN(x))
#endif
#ifndef RARRAY_LEN
# define RARRAY_LEN(x) RARRAY(x)->len
#endif
#ifndef RARRAY_PTR
# define RARRAY_PTR(x) RARRAY(x)->ptr
#endif
#ifndef RFLOAT_VALUE
# define RFLOAT_VALUE(x) RFLOAT(x)->value
#endif
#ifndef DOUBLE2NUM
# define DOUBLE2NUM(x) rb_float_new(x)
#endif
#ifndef RHASH_TBL
# define RHASH_TBL(x) (RHASH(x)->tbl)
#endif
#ifndef RHASH_ITER_LEV
# define RHASH_ITER_LEV(x) (RHASH(x)->iter_lev)
#endif
#ifndef RHASH_IFNONE
# define RHASH_IFNONE(x) (RHASH(x)->ifnone)
#endif
#ifndef RHASH_SIZE
# define RHASH_SIZE(x) (RHASH(x)->tbl->num_entries)
#endif
#ifndef RHASH_EMPTY_P
# define RHASH_EMPTY_P(x) (RHASH_SIZE(x) == 0)
#endif
#ifndef RSTRUCT_LEN
# define RSTRUCT_LEN(x) RSTRUCT(x)->len
#endif
#ifndef RSTRUCT_PTR
# define RSTRUCT_PTR(x) RSTRUCT(x)->ptr
#endif
#ifndef RTYPEDDATA_P
# define RTYPEDDATA_P(x) (TYPE(x) != T_DATA)
#endif



/*
 * The following macros are used for providing the correct type of a
 * function pointer to the Ruby C API.
 * Starting with Ruby 2.7 (corresponding to RB_METHOD_DEFINITION_DECL being
 * defined) these macros act transparently due to Ruby's moving away from
 * ANYARGS and instead employing strict function signatures.
 *
 * Note: In case of C (not C++) the macros are transparent even before
 * Ruby 2.7 due to the fact that the Ruby C API used function declarators
 * with empty parentheses, which allows for an unspecified number of
 * arguments.
 *
 * PROTECTFUNC(f) is used for the function pointer argument of the Ruby
 * C API function rb_protect().
 *
 * VALUEFUNC(f) is used for the function pointer argument(s) of Ruby C API
 * functions like rb_define_method() and rb_define_singleton_method().
 *
 * VOIDFUNC(f) is used to typecast a C function that implements either
 * the "mark" or "free" stuff for a Ruby Data object, so that it can be
 * passed as an argument to Ruby C API functions like Data_Wrap_Struct()
 * and Data_Make_Struct().
 *
 * SWIG_RUBY_VOID_ANYARGS_FUNC(f) is used for the function pointer
 * argument(s) of Ruby C API functions like rb_define_virtual_variable().
 *
 * SWIG_RUBY_INT_ANYARGS_FUNC(f) is used for the function pointer
 * argument(s) of Ruby C API functions like st_foreach().
 */
#if defined(__cplusplus) && !defined(RB_METHOD_DEFINITION_DECL)
#  define PROTECTFUNC(f) ((VALUE (*)(VALUE)) f)
#  define VALUEFUNC(f) ((VALUE (*)(ANYARGS)) f)
#  define VOIDFUNC(f) ((RUBY_DATA_FUNC) f)
#  define SWIG_RUBY_VOID_ANYARGS_FUNC(f) ((void (*)(ANYARGS))(f))
#  define SWIG_RUBY_INT_ANYARGS_FUNC(f) ((int (*)(ANYARGS))(f))
#else
#  define PROTECTFUNC(f) (f)
#  define VALUEFUNC(f) (f)
#  define VOIDFUNC(f) (f)
#  define SWIG_RUBY_VOID_ANYARGS_FUNC(f) (f)
#  define SWIG_RUBY_INT_ANYARGS_FUNC(f) (f)
#endif

/* Don't use for expressions have side effect */
#ifndef RB_STRING_VALUE
#define RB_STRING_VALUE(s) (TYPE(s) == T_STRING ? (s) : (*(volatile VALUE *)&(s) = rb_str_to_str(s)))
#endif
#ifndef StringValue
#define StringValue(s) RB_STRING_VALUE(s)
#endif
#ifndef StringValuePtr
#define StringValuePtr(s) RSTRING_PTR(RB_STRING_VALUE(s))
#endif
#ifndef StringValueLen
#define StringValueLen(s) RSTRING_LEN(RB_STRING_VALUE(s))
#endif
#ifndef SafeStringValue
#define SafeStringValue(v) do {\
    StringValue(v);\
    rb_check_safe_str(v);\
} while (0)
#endif

#ifndef HAVE_RB_DEFINE_ALLOC_FUNC
#define rb_define_alloc_func(klass, func) rb_define_singleton_method((klass), "new", VALUEFUNC((func)), -1)
#define rb_undef_alloc_func(klass) rb_undef_method(CLASS_OF((klass)), "new")
#endif

static VALUE _mSWIG = Qnil;
