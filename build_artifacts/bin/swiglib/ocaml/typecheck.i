/* -----------------------------------------------------------------------------
 * typecheck.i
 *
 * Typechecking rules
 * ----------------------------------------------------------------------------- */

%typecheck(SWIG_TYPECHECK_INT8) char, signed char, const char &, const signed char & {
  if( !Is_block($input) ) $1 = 0;
  else {
      switch( SWIG_Tag_val($input) ) {
      case C_char: $1 = 1; break;
      default: $1 = 0; break;
      }
  }
}

%typecheck(SWIG_TYPECHECK_UINT8) unsigned char, const unsigned char & {
  if( !Is_block($input) ) $1 = 0;
  else {
      switch( SWIG_Tag_val($input) ) {
      case C_uchar: $1 = 1; break;
      default: $1 = 0; break;
      }
  }
}

%typecheck(SWIG_TYPECHECK_INT16) short, signed short, const short &, const signed short &, wchar_t {
  if( !Is_block($input) ) $1 = 0;
  else {
      switch( SWIG_Tag_val($input) ) {
      case C_short: $1 = 1; break;
      default: $1 = 0; break;
      }
  }
}

%typecheck(SWIG_TYPECHECK_UINT16) unsigned short, const unsigned short & {
  if( !Is_block($input) ) $1 = 0;
  else {
      switch( SWIG_Tag_val($input) ) {
      case C_ushort: $1 = 1; break;
      default: $1 = 0; break;
      }
  }
}

// XXX arty 
// Will move enum SWIGTYPE later when I figure out what to do with it...

%typecheck(SWIG_TYPECHECK_INT32) int, signed int, const int &, const signed int &, enum SWIGTYPE {
  if( !Is_block($input) ) $1 = 0;
  else {
      switch( SWIG_Tag_val($input) ) {
      case C_int: $1 = 1; break;
      default: $1 = 0; break;
      }
  }
}

%typecheck(SWIG_TYPECHECK_UINT32) unsigned int, const unsigned int & {
  if( !Is_block($input) ) $1 = 0;
  else {
      switch( SWIG_Tag_val($input) ) {
      case C_uint: $1 = 1; break;
      case C_int32: $1 = 1; break;
      default: $1 = 0; break;
      }
  }
}

%typecheck(SWIG_TYPECHECK_INT64)
  long, signed long, unsigned long,
  long long, signed long long, unsigned long long,
  const long &, const signed long &, const unsigned long &,
  const long long &, const signed long long &, const unsigned long long &,
  size_t, const size_t &
{
  if( !Is_block($input) ) $1 = 0;
  else {
      switch( SWIG_Tag_val($input) ) {
      case C_int64: $1 = 1; break;
      default: $1 = 0; break;
      }
  }
}

%typecheck(SWIG_TYPECHECK_BOOL) bool, const bool & {
  if( !Is_block($input) ) $1 = 0;
  else {
      switch( SWIG_Tag_val($input) ) {
      case C_bool: $1 = 1; break;
      default: $1 = 0; break;
      }
  }
}

%typecheck(SWIG_TYPECHECK_FLOAT) float, const float & {
  if( !Is_block($input) ) $1 = 0;
  else {
      switch( SWIG_Tag_val($input) ) {
      case C_float: $1 = 1; break;
      default: $1 = 0; break;
      }
  }  
}

%typecheck(SWIG_TYPECHECK_DOUBLE) double, const double & {
  if( !Is_block($input) ) $1 = 0;
  else {
      switch( SWIG_Tag_val($input) ) {
      case C_double: $1 = 1; break;
      default: $1 = 0; break;
      }
  }  
}

%typecheck(SWIG_TYPECHECK_STRING) char * {
  if( !Is_block($input) ) $1 = 0;
  else {
      switch( SWIG_Tag_val($input) ) {
      case C_string: $1 = 1; break;
      case C_ptr: {
	swig_type_info *typeinfo = 
	    (swig_type_info *)(long)SWIG_Int64_val(SWIG_Field($input,1));
	$1 = SWIG_TypeCheck("char *",typeinfo) ||
	     SWIG_TypeCheck("signed char *",typeinfo) ||
	     SWIG_TypeCheck("unsigned char *",typeinfo) ||
	     SWIG_TypeCheck("const char *",typeinfo) ||
	     SWIG_TypeCheck("const signed char *",typeinfo) ||
	     SWIG_TypeCheck("const unsigned char *",typeinfo) ||
	     SWIG_TypeCheck("std::string",typeinfo);
      } break;
      default: $1 = 0; break;
      }
  }    
}

%typecheck(SWIG_TYPECHECK_POINTER) SWIGTYPE *, SWIGTYPE &, SWIGTYPE &&, SWIGTYPE [] {
  if (!Is_block($input) || !(SWIG_Tag_val($input) == C_obj || SWIG_Tag_val($input) == C_ptr)) {
    $1 = 0;
  } else {
    void *ptr;
    $1 = !caml_ptr_val_internal($input, &ptr, $descriptor);
  }
}

%typecheck(SWIG_TYPECHECK_POINTER) SWIGTYPE {
  swig_type_info *typeinfo;
  if (!Is_block($input)) {
    $1 = 0;
  } else {
    switch (SWIG_Tag_val($input)) {
      case C_obj: {
        void *ptr;
        $1 = !caml_ptr_val_internal($input, &ptr, $&1_descriptor);
        break;
      }
      case C_ptr: {
        typeinfo = (swig_type_info *)SWIG_Int64_val(SWIG_Field($input, 1));
        $1 = SWIG_TypeCheck("$1_type", typeinfo) != NULL;
        break;
      }
      default: $1 = 0; break;
    }
  }
}

%typecheck(SWIG_TYPECHECK_VOIDPTR) void * {
  void *ptr;
  $1 = !caml_ptr_val_internal($input, &ptr, 0);
}

%typecheck(SWIG_TYPECHECK_SWIGOBJECT) CAML_VALUE "$1 = 1;"

/* ------------------------------------------------------------
 * Exception handling
 * ------------------------------------------------------------ */

%typemap(throws) int, 
                  long, 
                  short, 
                  unsigned int, 
                  unsigned long, 
                  unsigned short {
  char error_msg[256];
  sprintf(error_msg, "C++ $1_type exception thrown, value: %d", $1);
  SWIG_OCamlThrowException(SWIG_OCamlRuntimeException, error_msg);
}

%typemap(throws) SWIGTYPE, SWIGTYPE &, SWIGTYPE &&, SWIGTYPE *, SWIGTYPE [], SWIGTYPE [ANY] {
  (void)$1;
  SWIG_OCamlThrowException(SWIG_OCamlRuntimeException, "C++ $1_type exception thrown");
}

%typemap(throws) char * {
  SWIG_OCamlThrowException(SWIG_OCamlRuntimeException, $1);
}
