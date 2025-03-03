%insert(mli) %{
type _value = c_obj
%}

%insert(ml) %{
type _value = c_obj
%}

%define %array_tmap_out(type,what,out_f)
%typemap(type) what [ANY] {
    int i;
    $result = caml_array_new($1_dim0);
    for( i = 0; i < $1_dim0; i++ ) {
	caml_array_set($result,i,out_f($1[i]));
    }
}
%enddef

%define %array_tmap_in(type,what,in_f)
%typemap(type) what [ANY] {
    int i;
    $1 = ($*1_type *)malloc( $1_size );
    for( i = 0; i < $1_dim0 && i < caml_array_len($input); i++ ) {
	$1[i] = in_f(caml_array_nth($input,i));
    }
}

%typemap(free) what [ANY] {
    free( (void *)$1 );
}
%enddef

%define %make_simple_array_typemap(type,out_f,in_f)
%array_tmap_out(out,type,out_f);
%array_tmap_out(varout,type,out_f);
%array_tmap_out(directorin,type,out_f);

%array_tmap_in(in,type,in_f);
%array_tmap_in(varin,type,in_f);
%array_tmap_in(directorout,type,in_f);
%enddef

%make_simple_array_typemap(bool,caml_val_bool,caml_long_val);
%make_simple_array_typemap(short,caml_val_short,caml_long_val);
%make_simple_array_typemap(unsigned short,caml_val_ushort,caml_long_val);
%make_simple_array_typemap(int,caml_val_int,caml_long_val);
%make_simple_array_typemap(unsigned int,caml_val_uint,caml_long_val);
%make_simple_array_typemap(long,caml_val_long,caml_long_val);
%make_simple_array_typemap(unsigned long,caml_val_ulong,caml_long_val);
%make_simple_array_typemap(size_t,caml_val_int,caml_long_val);
%make_simple_array_typemap(float,caml_val_float,caml_double_val);
%make_simple_array_typemap(double,caml_val_double,caml_double_val);

#ifdef __cplusplus
%typemap(in) SWIGTYPE [] {
    int i;

    $1 = new $*1_type [$1_dim0];
    for( i = 0; i < $1_dim0 && i < caml_array_len($input); i++ ) {
	$1[i] = *(($*1_ltype *) 
		caml_ptr_val(caml_array_nth($input,i),
			     $*1_descriptor)) ;
    }
}
#else
%typemap(in) SWIGTYPE [] {
    int i;

    $1 = ($*1_type *)malloc( $1_size );
    for( i = 0; i < $1_dim0 && i < caml_array_len($input); i++ ) {
	$1[i] = *(($*1_ltype)
		caml_ptr_val(caml_array_nth($input),
			     $*1_descriptor));
    }
}
#endif

%typemap(out) SWIGTYPE [] {
    int i;
    const CAML_VALUE *fromval = caml_named_value("create_$ntype_from_ptr");
    $result = caml_array_new($1_dim0);

    for( i = 0; i < $1_dim0; i++ ) {
	if( fromval ) {
	    caml_array_set 
		($result,
		 i,
		 caml_callback(*fromval,caml_val_ptr((void *)&$1[i],$*1_descriptor)));
	} else {
	    caml_array_set
		($result,
		 i,
		 caml_val_ptr ((void *)&$1[i],$&1_descriptor));
	}
    }
}

%typemap(in) enum SWIGTYPE [] {
    int i;

    $1 = ($*1_type *)malloc( $1_size );
    for( i = 0; i < $1_dim0 && i < caml_array_len($input); i++ ) {
	$1[i] = ($type)
		caml_long_val_full(caml_array_nth($input),
			           "$type_marker");
    }
}

%typemap(out) enum SWIGTYPE [] {
    int i;
    $result = caml_array_new($1_dim0);

    for( i = 0; i < $1_dim0; i++ ) {
	    caml_array_set 
		($result,
		 i,
		 caml_callback2(*caml_named_value(SWIG_MODULE "_int_to_enum"),
			   *caml_named_value("$type_marker"),
			   Val_int($1[i])));
    }
}

#ifdef __cplusplus
%typemap(freearg) SWIGTYPE [ANY] {
    delete [] $1;
}
#else
%typemap(freearg) SWIGTYPE [ANY] {
    free( (void *)$1 );
}
#endif
