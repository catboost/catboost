/* -----------------------------------------------------------------------------
 * ocamlrundec.swg
 *
 * Ocaml runtime code -- declarations
 * ----------------------------------------------------------------------------- */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
#define SWIGEXT extern "C"
SWIGEXT {
#else
#define SWIGEXT 
#endif
#define value caml_value_t
#define CAML_VALUE caml_value_t
#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/callback.h>
#include <caml/fail.h>
#include <caml/misc.h>

#if defined(CAMLassert)
/* Both this macro and version.h were introduced in version 4.02.0 */
#include <caml/version.h>
#else
#define OCAML_VERSION 0 /* Unknown, but < 40200 */
#endif

#define caml_array_set swig_caml_array_set

/* Adapted from memory.h and mlvalues.h */

#define SWIG_CAMLlocal1(x) \
  caml_value_t x = 0; \
  CAMLxparam1 (x)

#define SWIG_CAMLlocal2(x, y) \
  caml_value_t x = 0, y = 0; \
  CAMLxparam2 (x, y)

#define SWIG_CAMLlocal3(x, y, z) \
  caml_value_t x = 0, y = 0, z = 0; \
  CAMLxparam3 (x, y, z)

#define SWIG_CAMLlocal4(x, y, z, t) \
  caml_value_t x = 0, y = 0, z = 0, t = 0; \
  CAMLxparam4 (x, y, z, t)

#define SWIG_CAMLlocal5(x, y, z, t, u) \
  caml_value_t x = 0, y = 0, z = 0, t = 0, u = 0; \
  CAMLxparam5 (x, y, z, t, u)

#define SWIG_CAMLlocalN(x, size) \
  caml_value_t x [(size)] = { 0, /* 0, 0, ... */ }; \
  CAMLxparamN (x, (size))

#define SWIG_Field(x, i) (((caml_value_t *)(x)) [i])           /* Also an l-value. */
#define SWIG_Store_field(block, offset, val) do{ \
  mlsize_t caml__temp_offset = (offset); \
  caml_value_t caml__temp_val = (val); \
  caml_modify (&SWIG_Field ((block), caml__temp_offset), caml__temp_val); \
}while(0)

#define SWIG_Data_custom_val(v) ((void *) &SWIG_Field((v), 1))
#ifdef ARCH_BIG_ENDIAN
#define SWIG_Tag_val(val) (((unsigned char *) (val)) [-1])
                                                 /* Also an l-value. */
#define SWIG_Tag_hp(hp) (((unsigned char *) (hp)) [sizeof(caml_value_t)-1])
                                                 /* Also an l-value. */
#else
#define SWIG_Tag_val(val) (((unsigned char *) (val)) [-sizeof(caml_value_t)])
                                                 /* Also an l-value. */
#define SWIG_Tag_hp(hp) (((unsigned char *) (hp)) [0])
                                                 /* Also an l-value. */
#endif

#ifdef CAMLreturn0
#undef CAMLreturn0
#endif
#define CAMLreturn0 do{ \
  caml_local_roots = caml__frame; \
  return; \
}while (0)

#ifdef CAMLreturn
#undef CAMLreturn
#endif
#define CAMLreturn(result) do{ \
  caml_value_t caml__temp_result = (result); \
  caml_local_roots = caml__frame; \
  return (caml__temp_result); \
}while(0)

#define CAMLreturn_type(result) do{ \
  caml_local_roots = caml__frame; \
  return result; \
}while(0)

#ifdef CAMLnoreturn
#undef CAMLnoreturn
#endif
#define CAMLnoreturn ((void) caml__frame)


#ifndef ARCH_ALIGN_INT64
#if OCAML_VERSION >= 40300
#define SWIG_Int64_val(v) (*((int64_t *) SWIG_Data_custom_val(v)))
#else
#define SWIG_Int64_val(v) (*((int64 *) SWIG_Data_custom_val(v)))
#endif
#else
#if OCAML_VERSION >= 40300
CAMLextern int64_t Int64_val(caml_value_t v);
#else
CAMLextern int64 Int64_val(caml_value_t v);
#endif
#define SWIG_Int64_val(v) Int64_val(v)
#endif

#define SWIG_NewPointerObj(p,type,flags) caml_val_ptr(p,type)
#define SWIG_GetModule(clientdata) SWIG_Ocaml_GetModule(clientdata)
#define SWIG_SetModule(clientdata, pointer) SWIG_Ocaml_SetModule(pointer)

typedef enum {
  SWIG_OCamlArithmeticException,
  SWIG_OCamlDirectorPureVirtual,
  SWIG_OCamlOutOfMemoryError,
  SWIG_OCamlOverflowException,
  SWIG_OCamlIllegalArgumentException,
  SWIG_OCamlIndexOutOfBoundsException,
  SWIG_OCamlRuntimeException,
  SWIG_OCamlSystemException,
  SWIG_OCamlUnknownError
} SWIG_OCamlExceptionCodes;

SWIGINTERN void SWIG_OCamlThrowException(SWIG_OCamlExceptionCodes code, const char *msg) {
  CAMLparam0();
  SWIG_CAMLlocal1(str);

  switch (code) {
  case SWIG_OCamlIllegalArgumentException:
    caml_invalid_argument(msg);
    break;
  case SWIG_OCamlSystemException:
    str = caml_copy_string(msg);
    caml_raise_sys_error(str);
    break;
  case SWIG_OCamlArithmeticException:
  case SWIG_OCamlIndexOutOfBoundsException:
  case SWIG_OCamlOutOfMemoryError:
  case SWIG_OCamlOverflowException:
  case SWIG_OCamlRuntimeException:
  case SWIG_OCamlUnknownError:
  default:
    caml_failwith(msg);
    break;
  }
  CAMLreturn0;
}

#define SWIG_contract_assert(expr, msg) if(!(expr)) {SWIG_OCamlThrowException(SWIG_OCamlRuntimeException, msg);}

    SWIGINTERN int
    SWIG_GetPtr(void *source, void **result, swig_type_info *type, swig_type_info *result_type);

    SWIGINTERN CAML_VALUE caml_list_nth( CAML_VALUE lst, int n );
    SWIGINTERN CAML_VALUE caml_list_append( CAML_VALUE lst, CAML_VALUE elt );
    SWIGINTERN int caml_list_length( CAML_VALUE lst );
    SWIGINTERN CAML_VALUE caml_array_new( int n );
    SWIGINTERN void caml_array_set( CAML_VALUE arr, int n, CAML_VALUE item );
    SWIGINTERN CAML_VALUE caml_array_nth( CAML_VALUE arr, int n );
    SWIGINTERN int caml_array_len( CAML_VALUE arr );

    SWIGINTERN CAML_VALUE caml_val_char( char c );
    SWIGINTERN CAML_VALUE caml_val_uchar( unsigned char c );

    SWIGINTERN CAML_VALUE caml_val_short( short s );
    SWIGINTERN CAML_VALUE caml_val_ushort( unsigned short s );
    
    SWIGINTERN CAML_VALUE caml_val_int( int x );
    SWIGINTERN CAML_VALUE caml_val_uint( unsigned int x );

    SWIGINTERN CAML_VALUE caml_val_long( long x );
    SWIGINTERN CAML_VALUE caml_val_ulong( unsigned long x );

    SWIGINTERN CAML_VALUE caml_val_float( float f );
    SWIGINTERN CAML_VALUE caml_val_double( double d );

    SWIGINTERN CAML_VALUE caml_val_ptr( void *p, swig_type_info *descriptor );

    SWIGINTERN CAML_VALUE caml_val_string( const char *str );
    SWIGINTERN CAML_VALUE caml_val_string_len( const char *str, int len );

    SWIGINTERN long caml_long_val( CAML_VALUE v );
    SWIGINTERN double caml_double_val( CAML_VALUE v );

    SWIGINTERN int caml_ptr_val_internal( CAML_VALUE v, void **out,
				      swig_type_info *descriptor );
    SWIGINTERN void *caml_ptr_val( CAML_VALUE v, swig_type_info *descriptor );

    SWIGINTERN char *caml_string_val( CAML_VALUE v );
    SWIGINTERN int caml_string_len( CAML_VALUE v );

#ifdef __cplusplus
}
#endif
