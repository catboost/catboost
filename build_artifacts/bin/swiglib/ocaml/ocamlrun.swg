/* -*-c-*- */

/* SWIG pointer structure */

#include <string.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

#define C_bool 0
#define C_char 1
#define C_uchar 2
#define C_short 3
#define C_ushort 4
#define C_int 5
#define C_uint 6
#define C_int32 7
#define C_int64 8
#define C_float 9
#define C_double 10
#define C_ptr 11
#define C_array 12
#define C_list 13
#define C_obj 14
#define C_string 15
#define C_enum 16
#define C_director_core 17


/* Cast a pointer if possible; returns 1 if successful */
    
    SWIGINTERN int
    SWIG_Cast (void *source, swig_type_info *source_type,
	       void **ptr, swig_type_info *dest_type)
    {
	if( !source ) { /* Special case for NULL.  This is a popular question
	     for other modules on the list, so I want an easy way out... */
	    *ptr = 0;
	    return 0;
	}

#ifdef TYPE_CAST_VERBOSE
	fprintf( stderr, "Trying to cast %s to %s\n", 
		 source_type ? source_type->str : "<none>",
		 dest_type ? dest_type->str : "<none>" );
#endif
	if (dest_type != source_type) {
	    /* We have a type mismatch.  Will have to look through our type
	       mapping table to figure out whether or not we can accept this
	       datatype. 
	       --
	       Ignore typechecks for void *.  Allow any conversion. */
	    if( !dest_type || !source_type || 
		!strcmp(dest_type->name,"_p_void") ||
		!strcmp(source_type->name,"_p_void") ) {
		*ptr = source;
		return 0;
	    } else {
		swig_cast_info *tc = 
		    SWIG_TypeCheckStruct(source_type, dest_type );
#ifdef TYPE_CAST_VERBOSE
		fprintf( stderr, "Typecheck -> %s\n",
			 tc ? tc->type->str : "<none>" );
#endif
		if( tc ) {
		    int newmemory = 0;
		    *ptr = SWIG_TypeCast(tc, source, &newmemory);
		    assert(!newmemory); /* newmemory handling not yet implemented */
		    return 0;
		} else
		    return -1;
	    }
	} else {
	    *ptr = source;
	    return 0;
	}
    }

/* Return 0 if successful. */
    SWIGINTERN int
    SWIG_GetPtr(void *inptr, void **outptr, 
		swig_type_info *intype, swig_type_info *outtype) {
	if (intype) {
	    return SWIG_Cast(inptr, intype,
			     outptr, outtype) == -1;
	} else {
	    *outptr = inptr;
	    return 0;
	}
    }

    SWIGINTERN void caml_print_list( CAML_VALUE v );

    SWIGINTERN void caml_print_val( CAML_VALUE v ) {
	switch( SWIG_Tag_val(v) ) {
	case C_bool:
	    if( Bool_val(SWIG_Field(v,0)) ) fprintf( stderr, "true " );
	    else fprintf( stderr, "false " );
	    break;
	case C_char:
	case C_uchar:
	    fprintf( stderr, "'%c' (\\%03d) ", 
		     (Int_val(SWIG_Field(v,0)) >= ' ' &&
		      Int_val(SWIG_Field(v,0)) < 127) ? Int_val(SWIG_Field(v,0)) : '.',
		     Int_val(SWIG_Field(v,0)) );
	    break;
	case C_short:
	case C_ushort:
	case C_int:
	    fprintf( stderr, "%d ", (int)caml_long_val(v) );
	    break;

	case C_uint:
	case C_int32:
	    fprintf( stderr, "%ud ", (unsigned int)caml_long_val(v) );
	    break;
	case C_int64:
	    fprintf( stderr, "%ld ", caml_long_val(v) );
	    break;
	case C_float:
	case C_double:
	    fprintf( stderr, "%f ", caml_double_val(v) );
	    break;

	case C_ptr:
	{
	    void *vout = 0;
	    swig_type_info *ty = (swig_type_info *)(long)SWIG_Int64_val(SWIG_Field(v,1));
	    caml_ptr_val_internal(v,&vout,0);
	    fprintf( stderr, "PTR(%p,%s) ", 
		     vout,
		     ty ? ty->name : "(null)" );
	}
	break;
	case C_array:
	{
	    unsigned int i;
	    for( i = 0; i < Wosize_val( SWIG_Field(v,0) ); i++ ) 
		caml_print_val( SWIG_Field(SWIG_Field(v,0),i) );
	}
	break;
	case C_list:
	    caml_print_list( SWIG_Field(v,0) );
	    break;
	case C_obj:
	    fprintf( stderr, "OBJ(%p) ", (void *)SWIG_Field(v,0) );
	    break;
	case C_string:
	{
	    void *cout;
	    caml_ptr_val_internal(v,&cout,0);
	    fprintf( stderr, "'%s' ", (char *)cout );
	} 
	break;
	}
    }

    SWIGINTERN void caml_print_list( CAML_VALUE v ) {
	CAMLparam1(v);
	while( v && Is_block(v) ) {
	    fprintf( stderr, "[ " );
	    caml_print_val( SWIG_Field(v,0) );
	    fprintf( stderr, "]\n" );
	    v = SWIG_Field(v,1);
	}
	CAMLreturn0;
    }

    SWIGINTERN CAML_VALUE caml_list_nth( CAML_VALUE lst, int n ) {
	CAMLparam1(lst);
	int i = 0;
	while( i < n && lst && Is_block(lst) ) {
	    i++; lst = SWIG_Field(lst,1);
	}
	if( lst == Val_unit ) CAMLreturn(Val_unit);
	else CAMLreturn(SWIG_Field(lst,0));
    }
    
    SWIGINTERN CAML_VALUE caml_list_append( CAML_VALUE lst, CAML_VALUE elt ) {
	CAMLparam2(lst,elt);
	SWIG_CAMLlocal3(v,vt,lh);
	lh = Val_unit;
	v = Val_unit;

	/* Appending C_void should have no effect */
	if( !Is_block(elt) ) return lst;

	while( lst && Is_block(lst) ) {
	    if( v && v != Val_unit ) {
		vt = caml_alloc_tuple(2);
		SWIG_Store_field(v,1,vt);
		v = vt;
	    } else {
		v = lh = caml_alloc_tuple(2);
	    }
	    SWIG_Store_field(v,0,SWIG_Field(lst,0));
	    lst = SWIG_Field(lst,1);
	}

	if( v && Is_block(v) ) {
	    vt = caml_alloc_tuple(2);
	    SWIG_Store_field(v,1,vt);
	    v = vt;
	} else {
	    v = lh = caml_alloc_tuple(2);
	}
	SWIG_Store_field(v,0,elt);
	SWIG_Store_field(v,1,Val_unit);

	CAMLreturn(lh);
    }

    SWIGINTERN int caml_list_length( CAML_VALUE lst ) {
	CAMLparam1(lst);
	int i = 0;
	while( lst && Is_block(lst) ) { i++; lst = SWIG_Field(lst,1); }
	CAMLreturn(i);
    }

    SWIGINTERN void caml_array_set( CAML_VALUE arr, int n, CAML_VALUE item ) {
	CAMLparam2(arr,item);
	SWIG_Store_field(SWIG_Field(arr,0),n,item);
	CAMLreturn0;
    }

    SWIGINTERN value caml_array_nth( CAML_VALUE arr, int n ) {
	CAMLparam1(arr);
	if( SWIG_Tag_val(arr) == C_array )
	    CAMLreturn(SWIG_Field(SWIG_Field(arr,0),n));
	else if( SWIG_Tag_val(arr) == C_list )
	    CAMLreturn(caml_list_nth(arr,0));
	else
	    caml_failwith("Need array or list");
    }

    SWIGINTERN int caml_array_len( CAML_VALUE arr ) {
	CAMLparam1(arr);
	if( SWIG_Tag_val(arr) == C_array )
	    CAMLreturn(Wosize_val(SWIG_Field(arr,0)));
	else if( SWIG_Tag_val(arr) == C_list )
	    CAMLreturn(caml_list_length(arr));
	else
	    caml_failwith("Need array or list");
    }

    SWIGINTERN CAML_VALUE caml_swig_alloc(int x,int y) {
	return caml_alloc(x,y);
    }

    SWIGINTERN value caml_array_new( int n ) {
	CAMLparam0();
	SWIG_CAMLlocal1(vv);
	vv = caml_swig_alloc(1,C_array);
	SWIG_Store_field(vv,0,caml_alloc_tuple(n));
	CAMLreturn(vv);
    }
    
    SWIGINTERN CAML_VALUE caml_val_bool( int b ) {
	CAMLparam0();
	SWIG_CAMLlocal1(bv);
	bv = caml_swig_alloc(1,C_bool);
	SWIG_Store_field(bv,0,Val_bool(b));
	CAMLreturn(bv);
    }

    SWIGINTERN CAML_VALUE caml_val_char( char c ) {
	CAMLparam0();
	SWIG_CAMLlocal1(cv);
	cv = caml_swig_alloc(1,C_char);
	SWIG_Store_field(cv,0,Val_int(c));
	CAMLreturn(cv);
    }

    SWIGINTERN CAML_VALUE caml_val_uchar( unsigned char uc ) {
	CAMLparam0();
	SWIG_CAMLlocal1(ucv);
	ucv = caml_swig_alloc(1,C_uchar);
	SWIG_Store_field(ucv,0,Val_int(uc));
	CAMLreturn(ucv);
    }

    SWIGINTERN CAML_VALUE caml_val_short( short s ) {
	CAMLparam0();
	SWIG_CAMLlocal1(sv);
	sv = caml_swig_alloc(1,C_short);
	SWIG_Store_field(sv,0,Val_int(s));
	CAMLreturn(sv);
    }

    SWIGINTERN CAML_VALUE caml_val_ushort( unsigned short us ) {
	CAMLparam0();
	SWIG_CAMLlocal1(usv);
	usv = caml_swig_alloc(1,C_ushort);
	SWIG_Store_field(usv,0,Val_int(us));
	CAMLreturn(usv);
    }

    SWIGINTERN CAML_VALUE caml_val_int( int i ) {
	CAMLparam0();
	SWIG_CAMLlocal1(iv);
	iv = caml_swig_alloc(1,C_int);
	SWIG_Store_field(iv,0,Val_int(i));
	CAMLreturn(iv);
    }

    SWIGINTERN CAML_VALUE caml_val_uint( unsigned int ui ) {
	CAMLparam0();
	SWIG_CAMLlocal1(uiv);
	uiv = caml_swig_alloc(1,C_int);
	SWIG_Store_field(uiv,0,Val_int(ui));
	CAMLreturn(uiv);
    }

    SWIGINTERN CAML_VALUE caml_val_long( long l ) {
	CAMLparam0();
	SWIG_CAMLlocal1(lv);
	lv = caml_swig_alloc(1,C_int64);
	SWIG_Store_field(lv,0,caml_copy_int64(l));
	CAMLreturn(lv);
    }

    SWIGINTERN CAML_VALUE caml_val_ulong( unsigned long ul ) {
	CAMLparam0();
	SWIG_CAMLlocal1(ulv);
	ulv = caml_swig_alloc(1,C_int64);
	SWIG_Store_field(ulv,0,caml_copy_int64(ul));
	CAMLreturn(ulv);
    }

    SWIGINTERN CAML_VALUE caml_val_float( float f ) {
	CAMLparam0();
	SWIG_CAMLlocal1(fv);
	fv = caml_swig_alloc(1,C_float);
	SWIG_Store_field(fv,0,caml_copy_double((double)f));
	CAMLreturn(fv);
    }

    SWIGINTERN CAML_VALUE caml_val_double( double d ) {
	CAMLparam0();
	SWIG_CAMLlocal1(fv);
	fv = caml_swig_alloc(1,C_double);
	SWIG_Store_field(fv,0,caml_copy_double(d));
	CAMLreturn(fv);
    }

    SWIGINTERN CAML_VALUE caml_val_ptr( void *p, swig_type_info *info ) {
	CAMLparam0();
	SWIG_CAMLlocal1(vv);
	vv = caml_swig_alloc(2,C_ptr);
	SWIG_Store_field(vv,0,caml_copy_int64((long)p));
	SWIG_Store_field(vv,1,caml_copy_int64((long)info));
	CAMLreturn(vv);
    }

    SWIGINTERN CAML_VALUE caml_val_string( const char *p ) {
	CAMLparam0();
	SWIG_CAMLlocal1(vv);
	if( !p ) CAMLreturn(caml_val_ptr( (void *)p, 0 ));
	vv = caml_swig_alloc(1,C_string);
	SWIG_Store_field(vv,0,caml_copy_string(p));
	CAMLreturn(vv);
    }

    SWIGINTERN CAML_VALUE caml_val_string_len( const char *p, int len ) {
	CAMLparam0();
	SWIG_CAMLlocal1(vv);
	if( !p || len < 0 ) CAMLreturn(caml_val_ptr( (void *)p, 0 ));
	vv = caml_swig_alloc(1,C_string);
	SWIG_Store_field(vv,0,caml_alloc_string(len));
	memcpy(String_val(SWIG_Field(vv,0)),p,len);
	CAMLreturn(vv);
    }

    #define caml_val_obj(v, name) caml_val_obj_helper(v, SWIG_TypeQuery((name)), name)
    SWIGINTERN CAML_VALUE caml_val_obj_helper( void *v, swig_type_info *type, char *name) {
	CAMLparam0();
	CAMLreturn(caml_callback2(*caml_named_value("caml_create_object_fn"),
			     caml_val_ptr(v,type),
			     caml_copy_string(name)));
    }

    SWIGINTERN long caml_long_val_full( CAML_VALUE v, const char *name ) {
	CAMLparam1(v);
	if( !Is_block(v) ) return 0;

	switch( SWIG_Tag_val(v) ) {
	case C_bool:
	case C_char:
	case C_uchar:
	case C_short:
	case C_ushort:
	case C_int:
	    CAMLreturn(Int_val(SWIG_Field(v,0)));
	case C_uint:
	case C_int32:
	    CAMLreturn(Int32_val(SWIG_Field(v,0)));
	case C_int64:
	    CAMLreturn((long)SWIG_Int64_val(SWIG_Field(v,0)));
	case C_float:
	case C_double:
	    CAMLreturn((long)Double_val(SWIG_Field(v,0)));
	case C_string:
	    CAMLreturn((long)String_val(SWIG_Field(v,0)));
	case C_ptr:
	    CAMLreturn((long)SWIG_Int64_val(SWIG_Field(SWIG_Field(v,0),0)));
	case C_enum: {
	    SWIG_CAMLlocal1(ret);
	    const CAML_VALUE *enum_to_int = caml_named_value(SWIG_MODULE "_enum_to_int");
	    if( !name ) caml_failwith( "Not an enum conversion" );
	    ret = caml_callback2(*enum_to_int,*caml_named_value(name),v);
	    CAMLreturn(caml_long_val(ret));
	}
	default:
	    caml_failwith("No conversion to int");
	}
    }

    SWIGINTERN long caml_long_val( CAML_VALUE v ) {
	return caml_long_val_full(v,0);
    }

    SWIGINTERN double caml_double_val( CAML_VALUE v ) {
	CAMLparam1(v);
	if( !Is_block(v) ) return 0.0;
	switch( SWIG_Tag_val(v) ) {
	case C_bool:
	case C_char:
	case C_uchar:
	case C_short:
	case C_ushort:
	case C_int:
	    CAMLreturn_type(Int_val(SWIG_Field(v,0)));
	case C_uint:
	case C_int32:
	    CAMLreturn_type(Int32_val(SWIG_Field(v,0)));
	case C_int64:
	    CAMLreturn_type(SWIG_Int64_val(SWIG_Field(v,0)));
	case C_float:
	case C_double:
	    CAMLreturn_type(Double_val(SWIG_Field(v,0)));
	default:
	    fprintf( stderr, "Unknown block tag %d\n", SWIG_Tag_val(v) );
	    caml_failwith("No conversion to double");
	}
    }

    SWIGINTERN int caml_ptr_val_internal( CAML_VALUE v, void **out,
					  swig_type_info *descriptor ) {
	CAMLparam1(v);
	void *outptr = NULL;
        swig_type_info *outdescr = NULL;
        static const CAML_VALUE *func_val = NULL;

	if( v == Val_unit ) {
	    *out = 0;
	    CAMLreturn_type(0);
	}
	if( !Is_block(v) ) return -1;
	switch( SWIG_Tag_val(v) ) {
	case C_obj:
	    if (!func_val) {
	        func_val = caml_named_value("caml_obj_ptr");
	    }
	    CAMLreturn_type(caml_ptr_val_internal(caml_callback(*func_val, v), out, descriptor));
	case C_string:
	    outptr = (void *)String_val(SWIG_Field(v,0));
	    break;
	case C_ptr:
	    outptr = (void *)(long)SWIG_Int64_val(SWIG_Field(v,0));
            outdescr = (swig_type_info *)(long)SWIG_Int64_val(SWIG_Field(v,1));
	    break;
	default:
	    *out = 0;
	    CAMLreturn_type(1);
	    break;
	}

	CAMLreturn_type(SWIG_GetPtr(outptr, out, outdescr, descriptor));
    }

    SWIGINTERN void *caml_ptr_val( CAML_VALUE v, swig_type_info *descriptor ) {
        CAMLparam0();
#ifdef TYPE_CAST_VERBOSE
	caml_print_val( v );
#endif
	void *out = NULL;
	if( !caml_ptr_val_internal( v, &out, descriptor ) )
	    CAMLreturn_type(out);
	else
	    caml_failwith( "No appropriate conversion found." );
    }

    SWIGINTERN char *caml_string_val( CAML_VALUE v ) {
	return (char *)caml_ptr_val( v, 0 );
    }

    SWIGINTERN int caml_string_len( CAML_VALUE v ) {
	switch( SWIG_Tag_val(v) ) {
	case C_string:
	    return caml_string_length(SWIG_Field(v,0));
	default:
	    return strlen((char *)caml_ptr_val(v,0));
	}
    }

    SWIGINTERN int caml_bool_check( CAML_VALUE v ) {
	CAMLparam1(v);
	
	if( !Is_block(v) ) return 0;
	
	switch( SWIG_Tag_val(v) ) {
	case C_bool:
	case C_ptr:
	case C_string:
	    CAMLreturn(1);
	default:
	    CAMLreturn(0);
	}
    }

    SWIGINTERN int caml_int_check( CAML_VALUE v ) {
	CAMLparam1(v);
	
	if( !Is_block(v) ) return 0;
	
	switch( SWIG_Tag_val(v) ) {
	case C_char:
	case C_uchar:
	case C_short:
	case C_ushort:
	case C_int:
	case C_uint:
	case C_int32:
	case C_int64:
	    CAMLreturn(1);

	default:
	    CAMLreturn(0);
	}
    }

    SWIGINTERN int caml_float_check( CAML_VALUE v ) {
	CAMLparam1(v);
	if( !Is_block(v) ) return 0;

	switch( SWIG_Tag_val(v) ) {
	case C_float:
	case C_double:
	    CAMLreturn(1);

	default:
	    CAMLreturn(0);
	}	
    }

    SWIGINTERN int caml_ptr_check( CAML_VALUE v ) {
	CAMLparam1(v);
	if( !Is_block(v) ) return 0;

	switch( SWIG_Tag_val(v) ) {
	case C_string:
	case C_ptr:
	case C_int64:
	    CAMLreturn(1);

	default:
	    CAMLreturn(0);
	}	
    }

    SWIGINTERN CAML_VALUE SWIG_Ocaml_ptr_to_val(const char *name, void *ptr, swig_type_info *descriptor) {
        CAMLparam0();
        SWIG_CAMLlocal1(result);

        const CAML_VALUE *fromval = caml_named_value(name);
        if (fromval) {
            result = caml_callback(*fromval, caml_val_ptr(ptr, descriptor));
        } else {
            result = caml_val_ptr(ptr, descriptor);
        }
        CAMLreturn(result);
    }

    static swig_module_info *SWIG_Ocaml_GetModule(void *SWIGUNUSEDPARM(clientdata)) {
      CAML_VALUE pointer;

      pointer = caml_callback(*caml_named_value("swig_find_type_info"), caml_val_int(0));
      if (Is_block(pointer) && SWIG_Tag_val(pointer) == C_ptr) {
        return (swig_module_info *)(void *)(long)SWIG_Int64_val(SWIG_Field(pointer,0));
      }
      return 0;
    }

    static void SWIG_Ocaml_SetModule(swig_module_info *pointer) {
      CAML_VALUE mod_pointer;

      mod_pointer = caml_val_ptr(pointer, NULL);
      caml_callback(*caml_named_value("swig_set_type_info"), mod_pointer);
    }

#ifdef __cplusplus
}
#endif
#undef value

