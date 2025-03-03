// Scilab fragments for primitive types
%include <sciprimtypes.swg>

%include <scienum.swg>

// Scilab object type
#define SWIG_Object int

#define %append_output(obj)         if (!SWIG_IsOK(SWIG_Scilab_SetOutput(pvApiCtx, obj))) return SWIG_ERROR
#define %set_constant(name, obj)    if (!SWIG_IsOK(SWIG_Scilab_SetOutput(pvApiCtx, obj))) return SWIG_ERROR // Name is managed by the function name
#define %raise(obj, type, desc)     SWIG_Scilab_Raise(obj, type, desc)
#define %set_output(obj)            if (!SWIG_IsOK(SWIG_Scilab_SetOutput(pvApiCtx, obj))) return SWIG_ERROR
#define %set_varoutput(obj)         if (!SWIG_IsOK(SWIG_Scilab_SetOutput(pvApiCtx, obj))) return SWIG_ERROR
#define %set_argoutput(obj)         if (!SWIG_IsOK(SWIG_Scilab_SetOutput(pvApiCtx, obj))) return SWIG_ERROR

// Include the unified typemap library
%include <typemaps/swigtypemaps.swg>

/* ---------------------------------------------------------------------------*/
/* Generic typmemaps                                                          */
/*                                                                            */
/* This typemap is used when Scilab does not store this type directly         */
/* For example, a 'float' is stored in Scilab as a 'double'                   */
/* So we read a 'double' in Scilab and cast it to a 'float'                   */
/* ---------------------------------------------------------------------------*/

%define %scilab_in_typemap_withcast(TYPEMAPTYPE, FRAGMENTNAME, CTYPE, TEMPTYPE, TEMPINIT)
%typemap(TYPEMAPTYPE, fragment="FRAGMENTNAME") CTYPE {
  TEMPTYPE tempValue = TEMPINIT;
  if(FRAGMENTNAME(pvApiCtx, $input, &tempValue, SWIG_Scilab_GetFuncName()) != SWIG_OK) {
    return SWIG_ERROR;
  }
  $1 = (CTYPE) tempValue;
}
%enddef
%define %scilab_inptr_typemap(TYPEMAPTYPE, FRAGMENTNAME, CTYPE)
%typemap(TYPEMAPTYPE, noblock=1, fragment="FRAGMENTNAME") CTYPE {
  if (FRAGMENTNAME(pvApiCtx, $input, %as_voidptrptr(&$1), SWIG_Scilab_GetFuncName()) != SWIG_OK) {
    return SWIG_ERROR;
  }
}
%enddef

%define %scilab_out_typemap(TYPEMAPTYPE, FRAGMENTNAME, CTYPE)
%typemap(TYPEMAPTYPE, noblock=1, fragment="FRAGMENTNAME") CTYPE {
  if (FRAGMENTNAME(pvApiCtx, $result, $1) != SWIG_OK) {
    return SWIG_ERROR;
  }
}
%enddef

%define %scilab_outptr_typemap(TYPEMAPTYPE, FRAGMENTNAME, CTYPE)
%typemap(TYPEMAPTYPE, noblock=1, fragment="FRAGMENTNAME") CTYPE {
  if (FRAGMENTNAME(pvApiCtx, $result, %as_voidptr($1)) != SWIG_OK) {
    return SWIG_ERROR;
  }
}
%enddef

%define %scilab_varout_typemap(TYPEMAPTYPE, FRAGMENTNAME, CTYPE)
%typemap(TYPEMAPTYPE, noblock=1, fragment="FRAGMENTNAME") CTYPE {
  if (FRAGMENTNAME(pvApiCtx, $result, $value) != SWIG_OK) {
    return SWIG_ERROR;
  }
}
%enddef

%define %scilab_varoutptr_typemap(TYPEMAPTYPE, FRAGMENTNAME, CTYPE)
%typemap(TYPEMAPTYPE, noblock=1, fragment="FRAGMENTNAME") CTYPE {
  if (FRAGMENTNAME(pvApiCtx, $result, %as_voidptr($value)) != SWIG_OK) {
    return SWIG_ERROR;
  }
}
%enddef

%define %scilab_in_typemap(TYPEMAPTYPE, FRAGMENTNAME, CTYPE)
%typemap(TYPEMAPTYPE, noblock=1, fragment="FRAGMENTNAME") CTYPE {
  if (FRAGMENTNAME(pvApiCtx, $input, &$1, SWIG_Scilab_GetFuncName()) != SWIG_OK) {
    return SWIG_ERROR;
  }
}
%enddef


/* ---------------------------------------------------------------------------*/
/* Array typmemaps                                                            */
/* ---------------------------------------------------------------------------*/

%include <sciarray.swg>


/* ---------------------------------------------------------------------------*/
/* Enum typemaps                                                              */
/* ---------------------------------------------------------------------------*/

%typemap(in, noblock=1, fragment=SWIG_AsVal_frag(Enum)) enum SWIGTYPE (int val) {
  if (SWIG_AsVal_dec(Enum)($input, &val) != SWIG_OK) {
    return SWIG_ERROR;
  }
  $1 = %static_cast(val, $1_ltype);
}

%typemap(out, fragment=SWIG_From_frag(Enum)) enum SWIGTYPE {
  if (SWIG_From_dec(Enum)($1) != SWIG_OK) {
    return SWIG_ERROR;
  }
}

/* ---------------------------------------------------------------------------*/
/* Typecheck typemaps                                                         */
/* ---------------------------------------------------------------------------*/

%define %scilab_typecheck_generic(PRECEDENCE, TYPE_CHECK_FUNCTION, TYPE)
%typecheck(PRECEDENCE) TYPE {
  int *piAddrVar = NULL;
  SciErr sciErr = getVarAddressFromPosition(pvApiCtx, $input, &piAddrVar);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return SWIG_ERROR;
  }
  $1 = TYPE_CHECK_FUNCTION(pvApiCtx, piAddrVar);
}
%enddef

%fragment("SWIG_Check_SciDoubleOrInt", "header") {
SWIGINTERN int
SWIG_Check_SciDoubleOrInt(void *pvApiCtx, SwigSciObject iVar, int iIntegerType) {
  int *piAddrVar = NULL;
  int ret = 0;
  SciErr sciErr = getVarAddressFromPosition(pvApiCtx, iVar, &piAddrVar);
  if (sciErr.iErr) {
    printError(&sciErr, 0);
    return 0;
  }
  ret = isIntegerType(pvApiCtx, piAddrVar);
  if (ret == 1) {
    int iPrec = 0;
    sciErr = getMatrixOfIntegerPrecision(pvApiCtx, piAddrVar, &iPrec);
    if (sciErr.iErr) {
      printError(&sciErr, 0);
      return 0;
    }
    ret = (iPrec == iIntegerType) ? 1 : 0;
  }
  else {
    ret = isDoubleType(pvApiCtx, piAddrVar);
  }
  return ret;
}
}

/* Scilab equivalent for C integers can be sci_intXX or sci_matrix */
%define %scilab_typecheck_integer(PRECEDENCE, INTTYPE, TYPE)
%typecheck(PRECEDENCE, fragment="SWIG_Check_SciDoubleOrInt") TYPE {
  $1 = SWIG_Check_SciDoubleOrInt(pvApiCtx, $input, INTTYPE);
}
%enddef

%define %scilab_typecheck_pointer(PRECEDENCE, TYPE)
%typecheck(PRECEDENCE) TYPE {
  $1 = SwigScilabCheckPtr(pvApiCtx, $input, $descriptor, SWIG_Scilab_GetFuncName());
}
%enddef


// Double (and Float) have priority over before Integer type.

// Primitive types
%scilab_typecheck_pointer(SWIG_TYPECHECK_VOIDPTR, SWIGTYPE *)
%scilab_typecheck_pointer(SWIG_TYPECHECK_POINTER, SWIGTYPE *)
%scilab_typecheck_generic(SWIG_TYPECHECK_BOOL, isBooleanType,    bool)
%scilab_typecheck_generic(16, isDoubleType,                      double)
%scilab_typecheck_generic(17, isDoubleType,                      float)
%scilab_typecheck_integer(SWIG_TYPECHECK_INT8, SCI_INT8,         signed char)
%scilab_typecheck_integer(SWIG_TYPECHECK_UINT8, SCI_UINT8,       unsigned char)
%scilab_typecheck_integer(SWIG_TYPECHECK_INT16, SCI_INT16,       short)
%scilab_typecheck_integer(SWIG_TYPECHECK_UINT16, SCI_UINT16,     unsigned short)
%scilab_typecheck_integer(SWIG_TYPECHECK_INT32, SCI_INT32,       int)
%scilab_typecheck_integer(SWIG_TYPECHECK_INT32, SCI_INT32,       long)
%scilab_typecheck_integer(SWIG_TYPECHECK_UINT32, SCI_UINT32,     unsigned int)
%scilab_typecheck_integer(SWIG_TYPECHECK_UINT32, SCI_UINT32,     unsigned long)
%scilab_typecheck_integer(SWIG_TYPECHECK_INT32, SCI_INT32,       enum SWIGTYPE)
%scilab_typecheck_generic(SWIG_TYPECHECK_CHAR, isStringType,     char)

// Arrays
%scilab_typecheck_generic(SWIG_TYPECHECK_BOOL_ARRAY, isBooleanType,  bool)
%scilab_typecheck_generic(1016, isDoubleType,                        double [ANY])
%scilab_typecheck_generic(1017, isDoubleType,                        float [ANY])
%scilab_typecheck_integer(SWIG_TYPECHECK_INT8_ARRAY, SCI_INT8,       signed char [ANY])
%scilab_typecheck_integer(1026, SCI_UINT8,                           unsigned char [ANY])
%scilab_typecheck_integer(SWIG_TYPECHECK_INT16_ARRAY, SCI_INT16,     short [ANY])
%scilab_typecheck_integer(1036, SCI_UINT16,                          unsigned short [ANY])
%scilab_typecheck_integer(SWIG_TYPECHECK_INT32_ARRAY, SCI_INT32,     int [ANY])
%scilab_typecheck_integer(SWIG_TYPECHECK_INT32_ARRAY, SCI_INT32,     long [ANY])
%scilab_typecheck_integer(1046, SCI_UINT32,                          unsigned int [ANY])
%scilab_typecheck_integer(1046, SCI_UINT32,                          unsigned long [ANY])
%scilab_typecheck_generic(SWIG_TYPECHECK_CHAR_ARRAY, isStringType,   char [ANY])
%scilab_typecheck_generic(SWIG_TYPECHECK_STRING_ARRAY, isStringType, char *[ANY])
%scilab_typecheck_generic(SWIG_TYPECHECK_STRING_ARRAY, isStringType, char **)


/* ---------------------------------------------------------------------------*/
/* %scilabconstcode() feature typemaps                                        */
/* ---------------------------------------------------------------------------*/

%typemap(scilabconstcode, fragment=SWIG_CreateScilabVariable_frag(double)) double
%{
  if (SWIG_CreateScilabVariable_double(pvApiCtx, "$result", $value) != SWIG_OK)
    return SWIG_ERROR;
%}

%typemap(scilabconstcode, fragment=SWIG_CreateScilabVariable_frag(int)) int
%{
  if (SWIG_CreateScilabVariable_int(pvApiCtx, "$result", $value) != SWIG_OK)
    return SWIG_ERROR;
%}

%typemap(scilabconstcode, fragment=SWIG_CreateScilabVariable_frag(uint)) unsigned int
%{
  if (SWIG_CreateScilabVariable_uint(pvApiCtx, "$result", $value) != SWIG_OK)
    return SWIG_ERROR;
%}

%typemap(scilabconstcode, fragment=SWIG_CreateScilabVariable_frag(int)) long
%{
  if (SWIG_CreateScilabVariable_int(pvApiCtx, "$result", $value) != SWIG_OK)
    return SWIG_ERROR;
%}

%typemap(scilabconstcode, fragment=SWIG_CreateScilabVariable_frag(uint)) unsigned long
%{
  if (SWIG_CreateScilabVariable_uint(pvApiCtx, "$result", $value) != SWIG_OK)
    return SWIG_ERROR;
%}

%typemap(scilabconstcode, fragment=SWIG_CreateScilabVariable_frag(char)) char
%{
  if (SWIG_CreateScilabVariable_char(pvApiCtx, "$result", $value) != SWIG_OK)
    return SWIG_ERROR;
%}

%typemap(scilabconstcode, fragment=SWIG_CreateScilabVariable_frag(charptr)) char *
%{
  if (SWIG_CreateScilabVariable_charptr(pvApiCtx, "$result", $value) != SWIG_OK)
    return SWIG_ERROR;
%}

%typemap(scilabconstcode, fragment=SWIG_CreateScilabVariable_frag(double)) enum SWIGTYPE
%{
  if (SWIG_CreateScilabVariable_double(pvApiCtx, "$result", $value) != SWIG_OK)
    return SWIG_ERROR;
%}


/* ---------------------------------------------------------------------------*/
/* Exception typmemaps                                                        */
/* ---------------------------------------------------------------------------*/

%include <sciexception.swg>
