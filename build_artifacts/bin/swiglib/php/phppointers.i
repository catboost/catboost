%define %pass_by_ref( TYPE, CONVERT_IN, CONVERT_OUT )
%typemap(in, byref=1) TYPE *REF ($*1_ltype tmp),
             TYPE &REF ($*1_ltype tmp)
%{
  /* First Check for SWIG wrapped type */
  if (Z_ISNULL($input)) {
      $1 = 0;
  } else if (Z_ISREF($input)) {
      /* Not swig wrapped type, so we check if it's a PHP reference type */
      CONVERT_IN(tmp, $*1_ltype, $input);
      $1 = &tmp;
  } else {
      SWIG_PHP_Error(E_ERROR, SWIG_PHP_Arg_Error_Msg($argnum, Expected a reference));
  }
%}
%typemap(argout) TYPE *REF,
                 TYPE &REF
%{
  if (Z_ISREF($input)) {
    CONVERT_OUT(Z_REFVAL($input), tmp$argnum);
  }
%}
%enddef

%pass_by_ref( size_t, CONVERT_INT_IN, ZVAL_LONG );

%pass_by_ref( signed int, CONVERT_INT_IN, ZVAL_LONG );
%pass_by_ref( int, CONVERT_INT_IN, ZVAL_LONG );
%pass_by_ref( unsigned int, CONVERT_INT_IN, ZVAL_LONG );

%pass_by_ref( signed short, CONVERT_INT_IN, ZVAL_LONG );
%pass_by_ref( short, CONVERT_INT_IN, ZVAL_LONG );
%pass_by_ref( unsigned short, CONVERT_INT_IN, ZVAL_LONG );

%pass_by_ref( signed long, CONVERT_INT_IN, ZVAL_LONG );
%pass_by_ref( long, CONVERT_INT_IN, ZVAL_LONG );
%pass_by_ref( unsigned long, CONVERT_INT_IN, ZVAL_LONG );

%pass_by_ref( signed char, CONVERT_INT_IN, ZVAL_LONG );
%pass_by_ref( char, CONVERT_CHAR_IN, ZVAL_STRING );
%pass_by_ref( unsigned char, CONVERT_INT_IN, ZVAL_LONG );

%pass_by_ref( float, CONVERT_FLOAT_IN, ZVAL_DOUBLE );
%pass_by_ref( double, CONVERT_FLOAT_IN, ZVAL_DOUBLE );

%pass_by_ref( char *, CONVERT_CHAR_IN, ZVAL_STRING );
