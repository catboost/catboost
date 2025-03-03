/* -----------------------------------------------------------------------------
 * dswigtype.swg
 *
 * Typemaps for non-primitive types (C/C++ classes and structs).
 * ----------------------------------------------------------------------------- */

%typemap(ctype) SWIGTYPE "void *"
%typemap(imtype) SWIGTYPE "void*"
%typemap(dtype) SWIGTYPE "$&dclassname"

%typemap(ctype) SWIGTYPE [] "void *"
%typemap(imtype) SWIGTYPE [] "void*"
%typemap(dtype) SWIGTYPE [] "$dclassname"

%typemap(ctype) SWIGTYPE * "void *"
%typemap(imtype) SWIGTYPE * "void*"
%typemap(dtype, nativepointer="$dtype") SWIGTYPE * "$dclassname"

%typemap(ctype) SWIGTYPE & "void *"
%typemap(imtype) SWIGTYPE & "void*"
%typemap(dtype, nativepointer="$dtype") SWIGTYPE & "$dclassname"

%typemap(ctype) SWIGTYPE *const& "void *"
%typemap(imtype) SWIGTYPE *const& "void*"
%typemap(dtype) SWIGTYPE *const& "$*dclassname"

%typecheck(SWIG_TYPECHECK_POINTER)
    SWIGTYPE,
    SWIGTYPE *,
    SWIGTYPE &,
    SWIGTYPE [],
    SWIGTYPE *const&
  ""


/*
 * By-value conversion typemaps (parameter is converted to a pointer).
 */

%typemap(in, canthrow=1) SWIGTYPE ($&1_type argp)
%{ argp = ($&1_ltype)$input;
   if (!argp) {
     SWIG_DSetPendingException(SWIG_DIllegalArgumentException, "Attempt to dereference null $1_type");
     return $null;
   }
   $1 = *argp; %}

%typemap(out) SWIGTYPE
#ifdef __cplusplus
%{ $result = new $1_ltype((const $1_ltype &)$1); %}
#else
{
  $&1_ltype $1ptr = ($&1_ltype) malloc(sizeof($1_ltype));
  memmove($1ptr, &$1, sizeof($1_type));
  $result = $1ptr;
}
#endif

%typemap(directorin) SWIGTYPE
  "$input = (void *)new $1_ltype((const $1_ltype &)$1);"
%typemap(directorout) SWIGTYPE
%{ if (!$input) {
     SWIG_DSetPendingException(SWIG_DIllegalArgumentException, "Unexpected null return for type $1_type");
     return $null;
   }
   $result = *($&1_ltype)$input; %}

%typemap(ddirectorin) SWIGTYPE "new $&dclassname($winput, true)"
%typemap(ddirectorout) SWIGTYPE "$&dclassname.swigGetCPtr($dcall)"

%typemap(din) SWIGTYPE "$&dclassname.swigGetCPtr($dinput)"
%typemap(dout, excode=SWIGEXCODE) SWIGTYPE {
  $&dclassname ret = new $&dclassname($imcall, true);$excode
  return ret;
}


/*
 * Pointer conversion typemaps.
 */

%typemap(in) SWIGTYPE * "$1 = ($1_ltype)$input;"
%typemap(out) SWIGTYPE * "$result = (void *)$1;"

%typemap(directorin) SWIGTYPE *
  "$input = (void *) $1;"
%typemap(directorout, warning=SWIGWARN_TYPEMAP_DIRECTOROUT_PTR_MSG) SWIGTYPE *
  "$result = ($1_ltype)$input;"

%typemap(ddirectorin,
  nativepointer="cast($dtype)$winput"
) SWIGTYPE * "($winput is null) ? null : new $dclassname($winput, false)"

%typemap(ddirectorout,
  nativepointer="cast(void*)$dcall"
) SWIGTYPE * "$dclassname.swigGetCPtr($dcall)"

%typemap(din,
  nativepointer="cast(void*)$dinput"
) SWIGTYPE * "$dclassname.swigGetCPtr($dinput)"

%typemap(dout, excode=SWIGEXCODE,
  nativepointer="{\n  auto ret = cast($dtype)$imcall;$excode\n  return ret;\n}"
) SWIGTYPE * {
  void* cPtr = $imcall;
  $dclassname ret = (cPtr is null) ? null : new $dclassname(cPtr, $owner);$excode
  return ret;
}

// Use the same typemaps for const pointers.
%apply SWIGTYPE * { SWIGTYPE *const }


/*
 * Reference conversion typemaps.
 */

%typemap(in, canthrow=1) SWIGTYPE & %{ $1 = ($1_ltype)$input;
  if (!$1) {
    SWIG_DSetPendingException(SWIG_DIllegalArgumentException, "$1_type type is null");
    return $null;
  } %}
%typemap(out) SWIGTYPE & "$result = (void *)$1;"

%typemap(directorin) SWIGTYPE &
  "$input = ($1_ltype) &$1;"
%typemap(directorout, warning=SWIGWARN_TYPEMAP_DIRECTOROUT_PTR_MSG) SWIGTYPE &
%{ if (!$input) {
     SWIG_DSetPendingException(SWIG_DIllegalArgumentException, "Unexpected null return for type $1_type");
     return $null;
   }
   $result = ($1_ltype)$input; %}

%typemap(ddirectorin,
  nativepointer="cast($dtype)$winput"
) SWIGTYPE & "new $dclassname($winput, false)"
%typemap(ddirectorout,
  nativepointer="cast(void*)$dcall"
) SWIGTYPE & "$dclassname.swigGetCPtr($dcall)"

%typemap(din,
  nativepointer="cast(void*)$dinput"
) SWIGTYPE & "$dclassname.swigGetCPtr($dinput)"
%typemap(dout, excode=SWIGEXCODE,
  nativepointer="{\n  auto ret = cast($dtype)$imcall;$excode\n  return ret;\n}") SWIGTYPE & {
  $dclassname ret = new $dclassname($imcall, $owner);$excode
  return ret;
}


/*
 * Array conversion typemaps.
 */

%typemap(in) SWIGTYPE [] %{ $1 = ($1_ltype)$input; %}
%typemap(out) SWIGTYPE [] %{ $result = $1; %}

%typemap(din) SWIGTYPE [] "$dclassname.swigGetCPtr($dinput)"
%typemap(dout, excode=SWIGEXCODE) SWIGTYPE [] {
  void* cPtr = $imcall;
  $dclassname ret = (cPtr is null) ? null : new $dclassname(cPtr, $owner);$excode
  return ret;
}

// Treat references to arrays like references to a single element.
%apply SWIGTYPE & { SWIGTYPE ((&)[ANY]) }


/*
 * Pointer reference conversion typemaps.
 */

%typemap(in) SWIGTYPE *const& ($*1_ltype temp = 0)
%{ temp = ($*1_ltype)$input;
   $1 = ($1_ltype)&temp; %}
%typemap(out) SWIGTYPE *const&
%{ $result = (void *)*$1; %}

%typemap(din) SWIGTYPE *const& "$*dclassname.swigGetCPtr($dinput)"
%typemap(dout, excode=SWIGEXCODE) SWIGTYPE *const& {
  void* cPtr = $imcall;
  $*dclassname ret = (cPtr is null) ? null : new $*dclassname(cPtr, $owner);$excode
  return ret;
}
%typemap(directorin) SWIGTYPE *const&
  "$input = (void *) $1;"
%typemap(directorout, warning=SWIGWARN_TYPEMAP_DIRECTOROUT_PTR_MSG) SWIGTYPE *const&
%{ static $*1_ltype swig_temp;
   swig_temp = ($*1_ltype)$input;
   $result = &swig_temp; %}
%typemap(ddirectorin,
  nativepointer="cast($dtype)$winput"
) SWIGTYPE *const& "($winput is null) ? null : new $*dclassname($winput, false)"
%typemap(ddirectorout,
  nativepointer="cast(void*)$dcall"
) SWIGTYPE *const& "$*dclassname.swigGetCPtr($dcall)"

