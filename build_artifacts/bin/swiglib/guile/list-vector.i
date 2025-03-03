/* -----------------------------------------------------------------------------
 * list_vector.i
 *
 * Guile typemaps for converting between arrays and Scheme lists or vectors  
 * ----------------------------------------------------------------------------- */

/* Here is a macro that will define typemaps for converting between C
   arrays and Scheme lists or vectors when passing arguments to the C
   function.

   TYPEMAP_LIST_VECTOR_INPUT_OUTPUT(C_TYPE, SCM_TO_C, C_TO_SCM, SCM_TYPE)
   
   Supported calling conventions:

   func(int VECTORLENINPUT, [const] C_TYPE *VECTORINPUT)

       Scheme wrapper will take one argument, a vector.  A temporary C
       array of elements of type C_TYPE will be allocated and filled
       with the elements of the vectors, converted to C with the
       SCM_TO_C function.  Length and address of the array are passed
       to the C function.

       SCM_TYPE is used to describe the Scheme type of the elements in
       the Guile procedure documentation.
   
   func(int LISTLENINPUT, [const] C_TYPE *LISTINPUT)

       Likewise, but the Scheme wrapper will take one argument, a list.

   func(int *VECTORLENOUTPUT, C_TYPE **VECTOROUTPUT)

       Scheme wrapper will take no arguments.  Addresses of an integer
       and a C_TYPE * variable will be passed to the C function.  The
       C function is expected to return address and length of a
       freshly allocated array of elements of type C_TYPE through
       these pointers.  The elements of this array are converted to
       Scheme with the C_TO_SCM function and returned as a Scheme
       vector. 

       If the function has a void return value, the vector constructed
       by this typemap becomes the return value of the Scheme wrapper.
       Otherwise, the function returns multiple values.  (See
       the documentation on how to deal with multiple values.)

   func(int *LISTLENOUTPUT, C_TYPE **LISTOUTPUT)

       Likewise, but the Scheme wrapper will return a list instead of
       a vector.

   It is also allowed to use "size_t LISTLENINPUT" rather than "int
   LISTLENINPUT".  */

%define TYPEMAP_LIST_VECTOR_INPUT_WITH_EXPR(C_TYPE, SCM_TO_C_EXPR, SCM_TYPE)

  /* input */
     
     /* We make use of the new multi-dispatch typemaps here. */
     
     %typemap(in, doc="$NAME is a vector of " #SCM_TYPE " values")
       (int VECTORLENINPUT, C_TYPE *VECTORINPUT),
       (size_t VECTORLENINPUT, C_TYPE *VECTORINPUT)
     {
       SCM_VALIDATE_VECTOR($argnum, $input);
       $1 = scm_c_vector_length($input);
       if ($1 > 0) {
	 $1_ltype i;
	 $2 = (C_TYPE *) SWIG_malloc(sizeof(C_TYPE) * $1);
	 for (i = 0; i<$1; i++) {
	   SCM swig_scm_value = scm_vector_ref($input, scm_from_long(i));
	   $2[i] = SCM_TO_C_EXPR;
	 }
       }
       else $2 = NULL;
     }
	 
     %typemap(in, doc="$NAME is a list of " #SCM_TYPE " values")
       (int LISTLENINPUT, C_TYPE *LISTINPUT),
       (size_t LISTLENINPUT, C_TYPE *LISTINPUT)
     {
       SCM_VALIDATE_LIST($argnum, $input);
       $1 = scm_to_ulong(scm_length($input));
       if ($1 > 0) {
	 $1_ltype i;
	 SCM rest;
	 $2 = (C_TYPE *) SWIG_malloc(sizeof(C_TYPE) * $1);
	 for (i = 0, rest = $input;
	      i<$1;
	      i++, rest = SCM_CDR(rest)) {
	   SCM swig_scm_value = SCM_CAR(rest);
	   $2[i] = SCM_TO_C_EXPR;
	 }
       }
       else $2 = NULL;
     }

     /* Do not check for NULL pointers (override checks). */

     %typemap(check) (int VECTORLENINPUT, C_TYPE *VECTORINPUT),
                     (size_t VECTORLENINPUT, C_TYPE *VECTORINPUT),
                     (int LISTLENINPUT, C_TYPE *LISTINPUT),
                     (size_t LISTLENINPUT, C_TYPE *LISTINPUT)
       "/* no check for NULL pointer */";

     /* Discard the temporary array after the call. */

     %typemap(freearg) (int VECTORLENINPUT, C_TYPE *VECTORINPUT),
                     (size_t VECTORLENINPUT, C_TYPE *VECTORINPUT),
                     (int LISTLENINPUT, C_TYPE *LISTINPUT),
                     (size_t LISTLENINPUT, C_TYPE *LISTINPUT)
       {if ($2!=NULL) SWIG_free($2);}

%enddef

  /* output */

%define TYPEMAP_LIST_VECTOR_OUTPUT_WITH_EXPR(C_TYPE, C_TO_SCM_EXPR, SCM_TYPE)

     /* First we make temporary variables ARRAYLENTEMP and ARRAYTEMP,
	whose addresses we pass to the C function.  We ignore both
	arguments for Scheme. */

     %typemap(in,numinputs=0) (int *VECTORLENOUTPUT, C_TYPE **VECTOROUTPUT)
                        (int arraylentemp, C_TYPE *arraytemp),
                      (int *LISTLENOUTPUT, C_TYPE **LISTOUTPUT)
                        (int arraylentemp, C_TYPE *arraytemp),
		      (size_t *VECTORLENOUTPUT, C_TYPE **VECTOROUTPUT)
                        (size_t arraylentemp, C_TYPE *arraytemp),
                      (size_t *LISTLENOUTPUT, C_TYPE **LISTOUTPUT)
                        (size_t arraylentemp, C_TYPE *arraytemp)
     %{
       $1 = &arraylentemp;
       $2 = &arraytemp;
     %}

     /* In the ARGOUT typemaps, we convert the array into a vector or
        a list and append it to the results. */

     %typemap(argout, doc="$NAME (a vector of " #SCM_TYPE " values)") 
          (int *VECTORLENOUTPUT, C_TYPE **VECTOROUTPUT),
	  (size_t *VECTORLENOUTPUT, C_TYPE **VECTOROUTPUT)
     {
       $*1_ltype i;
       SCM res = scm_make_vector(scm_from_long(*$1),
				SCM_BOOL_F);
       for (i = 0; i<*$1; i++) {
	 C_TYPE swig_c_value = (*$2)[i];
	 SCM elt = C_TO_SCM_EXPR;
	 scm_vector_set_x(res, scm_from_long(i), elt);
       }
       SWIG_APPEND_VALUE(res);
     }

     %typemap(argout, doc="$NAME (a list of " #SCM_TYPE " values)")
          (int *LISTLENOUTPUT, C_TYPE **LISTOUTPUT),
	  (size_t *LISTLENOUTPUT, C_TYPE **LISTOUTPUT)
     {
       int i;
       SCM res = SCM_EOL;
       for (i = ((int)(*$1)) - 1; i>=0; i--) {
         C_TYPE swig_c_value = (*$2)[i];
	 SCM elt = C_TO_SCM_EXPR;
	 res = scm_cons(elt, res);
       }
       SWIG_APPEND_VALUE(res);
     }

     /* In the FREEARG typemaps, get rid of the C vector.  
        (This can be overridden if you want to keep the C vector.) */

     %typemap(freearg) 
          (int *VECTORLENOUTPUT, C_TYPE **VECTOROUTPUT),
	  (size_t *VECTORLENOUTPUT, C_TYPE **VECTOROUTPUT), 
	  (int *LISTLENOUTPUT, C_TYPE **LISTOUTPUT),
	  (size_t *LISTLENOUTPUT, C_TYPE **LISTOUTPUT)
     {
        if ((*$2)!=NULL) free(*$2);
     }

%enddef

%define TYPEMAP_LIST_VECTOR_INPUT_OUTPUT_WITH_EXPR(C_TYPE, SCM_TO_C_EXPR, C_TO_SCM_EXPR, SCM_TYPE)
  TYPEMAP_LIST_VECTOR_INPUT_WITH_EXPR(C_TYPE, SCM_TO_C_EXPR, SCM_TYPE)
  TYPEMAP_LIST_VECTOR_OUTPUT_WITH_EXPR(C_TYPE, C_TO_SCM_EXPR, SCM_TYPE)
%enddef

%define TYPEMAP_LIST_VECTOR_INPUT(C_TYPE, SCM_TO_C, SCM_TYPE)
  TYPEMAP_LIST_VECTOR_INPUT_WITH_EXPR
     (C_TYPE, SCM_TO_C(swig_scm_value), SCM_TYPE)
%enddef

%define TYPEMAP_LIST_VECTOR_OUTPUT(C_TYPE, C_TO_SCM, SCM_TYPE)
  TYPEMAP_LIST_VECTOR_OUTPUT_WITH_EXPR
     (C_TYPE, C_TO_SCM(swig_c_value), SCM_TYPE)
%enddef

%define TYPEMAP_LIST_VECTOR_INPUT_OUTPUT(C_TYPE, SCM_TO_C, C_TO_SCM, SCM_TYPE)
  TYPEMAP_LIST_VECTOR_INPUT_OUTPUT_WITH_EXPR
     (C_TYPE, SCM_TO_C(swig_scm_value), C_TO_SCM(swig_c_value), SCM_TYPE)
%enddef

/* We use the macro to define typemaps for some standard types. */

TYPEMAP_LIST_VECTOR_INPUT_OUTPUT(bool, scm_is_true, scm_from_bool, boolean);
TYPEMAP_LIST_VECTOR_INPUT_OUTPUT(char, SCM_CHAR, SCM_MAKE_CHAR, char);
TYPEMAP_LIST_VECTOR_INPUT_OUTPUT(unsigned char, SCM_CHAR, SCM_MAKE_CHAR, char);
TYPEMAP_LIST_VECTOR_INPUT_OUTPUT(int, scm_to_int, scm_from_long, integer);
TYPEMAP_LIST_VECTOR_INPUT_OUTPUT(short, scm_to_int, scm_from_long, integer);
TYPEMAP_LIST_VECTOR_INPUT_OUTPUT(long, scm_to_long, scm_from_long, integer);
TYPEMAP_LIST_VECTOR_INPUT_OUTPUT(ptrdiff_t, scm_to_long, scm_from_long, integer);
TYPEMAP_LIST_VECTOR_INPUT_OUTPUT(unsigned int, scm_to_ulong, scm_from_ulong, integer);
TYPEMAP_LIST_VECTOR_INPUT_OUTPUT(unsigned short, scm_to_ulong, scm_from_ulong, integer);
TYPEMAP_LIST_VECTOR_INPUT_OUTPUT(unsigned long, scm_to_ulong, scm_from_ulong, integer);
TYPEMAP_LIST_VECTOR_INPUT_OUTPUT(size_t, scm_to_ulong, scm_from_ulong, integer);
TYPEMAP_LIST_VECTOR_INPUT_OUTPUT(float, scm_to_double, scm_from_double, real);
TYPEMAP_LIST_VECTOR_INPUT_OUTPUT(double, scm_to_double, scm_from_double, real);
TYPEMAP_LIST_VECTOR_INPUT_OUTPUT(char *, SWIG_scm2str, SWIG_str02scm, string);
TYPEMAP_LIST_VECTOR_INPUT_OUTPUT(const char *, SWIG_scm2str, SWIG_str02scm, string);

/* For the char *, free all strings after converting */

     %typemap(freearg) 
          (int *VECTORLENOUTPUT, char ***VECTOROUTPUT),
	  (size_t *VECTORLENOUTPUT, char ***VECTOROUTPUT), 
	  (int *LISTLENOUTPUT, char ***LISTOUTPUT),
    (size_t *LISTLENOUTPUT, char ***LISTOUTPUT),
    (int *VECTORLENOUTPUT, const char ***VECTOROUTPUT),
    (size_t *VECTORLENOUTPUT, const char ***VECTOROUTPUT), 
    (int *LISTLENOUTPUT, const char ***LISTOUTPUT),
    (size_t *LISTLENOUTPUT, const char ***LISTOUTPUT)
    {
	 if ((*$2)!=NULL) {
	     int i;
	     for (i = 0; i < *$1; i++) {
		 if ((*$2)[i] != NULL) free((*$2)[i]);
	     }
	     free(*$2);
	 }
     }

%typemap(freearg) (int VECTORLENINPUT, char **VECTORINPUT),
    (size_t VECTORLENINPUT, char **VECTORINPUT), 
    (int LISTLENINPUT, char **LISTINPUT),
    (size_t LISTLENINPUT, char **LISTINPUT),
    (int VECTORLENINPUT, const char **VECTORINPUT),
    (size_t VECTORLENINPUT, const char **VECTORINPUT), 
    (int LISTLENINPUT, const char **LISTINPUT),
    (size_t LISTLENINPUT, const char **LISTINPUT)
{
    if (($2)!=NULL) {
	int i;
	for (i = 0; i< $1; i++)
	    if (($2)[i] != NULL) free(($2)[i]);
	free($2);
    }
}


/* Following is a macro that emits typemaps that are much more
   flexible.  (They are also messier.)  It supports multiple parallel
   lists and vectors (sharing one length argument each).

   TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_OUTPUT(C_TYPE, SCM_TO_C, C_TO_SCM, SCM_TYPE)
   
   Supported calling conventions:

   func(int PARALLEL_VECTORLENINPUT, [const] C_TYPE *PARALLEL_VECTORINPUT, ...)  or
   func([const] C_TYPE *PARALLEL_VECTORINPUT, ..., int PARALLEL_VECTORLENINPUT)

   func(int PARALLEL_LISTLENINPUT, [const] C_TYPE *PARALLEL_LISTINPUT, ...) or
   func([const] C_TYPE *PARALLEL_LISTINPUT, ..., int PARALLEL_LISTLENINPUT)

   func(int *PARALLEL_VECTORLENOUTPUT, C_TYPE **PARALLEL_VECTOROUTPUT, ...) or
   func(C_TYPE **PARALLEL_VECTOROUTPUT, int *PARALLEL_VECTORLENOUTPUT, ...)

   func(int *PARALLEL_LISTLENOUTPUT, C_TYPE **PARALLEL_LISTOUTPUT) or
   func(C_TYPE **PARALLEL_LISTOUTPUT, int *PARALLEL_LISTLENOUTPUT)

   It is also allowed to use "size_t PARALLEL_LISTLENINPUT" rather than "int
   PARALLEL_LISTLENINPUT".  */

%define TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_WITH_EXPR(C_TYPE, SCM_TO_C_EXPR, SCM_TYPE)

  /* input */
     
     /* Passing data is a little complicated here; just remember:
	IGNORE typemaps come first, then IN, then CHECK.  But if
	IGNORE is given, IN won't be used for this type.

	We need to "ignore" one of the parameters because there shall
	be only one argument on the Scheme side.  Here we only
	initialize the array length to 0 but save its address for a
	later change.  */
     
     %typemap(in,numinputs=0) int PARALLEL_VECTORLENINPUT (int *_global_vector_length),
		      size_t PARALLEL_VECTORLENINPUT (size_t *_global_vector_length)
     {		      
       $1 = 0;
       _global_vector_length = &$1;
     }

     %typemap(in,numinputs=0) int PARALLEL_LISTLENINPUT (int *_global_list_length),   
		      size_t PARALLEL_LISTLENINPUT (size_t *_global_list_length)
     {		      
       $1 = 0;
       _global_list_length = &$1;
     }

     /* All the work is done in IN. */

     %typemap(in, doc="$NAME is a vector of " #SCM_TYPE " values") 
		  C_TYPE *PARALLEL_VECTORINPUT,
		  const C_TYPE *PARALLEL_VECTORINPUT
     {
       SCM_VALIDATE_VECTOR($argnum, $input);
       *_global_vector_length = scm_c_vector_length($input);
       if (*_global_vector_length > 0) {
	 int i;
	 $1 = (C_TYPE *) SWIG_malloc(sizeof(C_TYPE)
			       * (*_global_vector_length));
	 for (i = 0; i<*_global_vector_length; i++) {
	   SCM swig_scm_value = scm_vector_ref($input, scm_from_long(i));
	   $1[i] = SCM_TO_C_EXPR;
	 }
       }
       else $1 = NULL;
     }
	 
     %typemap(in, doc="$NAME is a list of " #SCM_TYPE " values") 
		  C_TYPE *PARALLEL_LISTINPUT,
		  const C_TYPE *PARALLEL_LISTINPUT
     {
       SCM_VALIDATE_LIST($argnum, $input);
       *_global_list_length = scm_to_ulong(scm_length($input));
       if (*_global_list_length > 0) {
	 int i;
	 SCM rest;
	 $1 = (C_TYPE *) SWIG_malloc(sizeof(C_TYPE)
			       * (*_global_list_length));
	 for (i = 0, rest = $input;
	      i<*_global_list_length;
	      i++, rest = SCM_CDR(rest)) {
	   SCM swig_scm_value = SCM_CAR(rest);
	   $1[i] = SCM_TO_C_EXPR;
	 }
       }
       else $1 = NULL;
     }

     /* Don't check for NULL pointers (override checks). */

     %typemap(check) C_TYPE *PARALLEL_VECTORINPUT, 
		     const C_TYPE *PARALLEL_VECTORINPUT,
		     C_TYPE *PARALLEL_LISTINPUT, 
		     const C_TYPE *PARALLEL_LISTINPUT
       "/* no check for NULL pointer */";

     /* Discard the temporary array after the call. */

     %typemap(freearg) C_TYPE *PARALLEL_VECTORINPUT, 
		       const C_TYPE *PARALLEL_VECTORINPUT,
		       C_TYPE *PARALLEL_LISTINPUT, 
		       const C_TYPE *PARALLEL_LISTINPUT
       {if ($1!=NULL) SWIG_free($1);}

%enddef

%define TYPEMAP_PARALLEL_LIST_VECTOR_OUTPUT_WITH_EXPR(C_TYPE, C_TO_SCM_EXPR, SCM_TYPE)

  /* output */

     /* First we make a temporary variable ARRAYLENTEMP, use its
        address as the ...LENOUTPUT argument for the C function and
        "ignore" the ...LENOUTPUT argument for Scheme.  */

     %typemap(in,numinputs=0) int *PARALLEL_VECTORLENOUTPUT (int _global_arraylentemp),
		      size_t *PARALLEL_VECTORLENOUTPUT (size_t _global_arraylentemp),
		      int *PARALLEL_LISTLENOUTPUT   (int _global_arraylentemp),
		      size_t *PARALLEL_LISTLENOUTPUT   (size_t _global_arraylentemp)
       "$1 = &_global_arraylentemp;";

     /* We also need to ignore the ...OUTPUT argument. */

     %typemap(in,numinputs=0) C_TYPE **PARALLEL_VECTOROUTPUT (C_TYPE *arraytemp),
		      C_TYPE **PARALLEL_LISTOUTPUT   (C_TYPE *arraytemp)
       "$1 = &arraytemp;";

     /* In the ARGOUT typemaps, we convert the array into a vector or
        a list and append it to the results. */

     %typemap(argout, doc="$NAME (a vector of " #SCM_TYPE " values)") 
		      C_TYPE **PARALLEL_VECTOROUTPUT
     {
       int i;
       SCM res = scm_make_vector(scm_from_long(_global_arraylentemp),
				SCM_BOOL_F);
       for (i = 0; i<_global_arraylentemp; i++) {
         C_TYPE swig_c_value = (*$1)[i];
	 SCM elt = C_TO_SCM_EXPR;
	 scm_vector_set_x(res, scm_from_long(i), elt);
       }
       SWIG_APPEND_VALUE(res);
     }

     %typemap(argout, doc="$NAME (a list of " #SCM_TYPE " values)") 
		      C_TYPE **PARALLEL_LISTOUTPUT
     {
       int i;
       SCM res = SCM_EOL;
       if (_global_arraylentemp > 0) {
         for (i = _global_arraylentemp - 1; i>=0; i--) {
	   C_TYPE swig_c_value = (*$1)[i];	 
	   SCM elt = C_TO_SCM_EXPR;
	   res = scm_cons(elt, res);
         }
       }
       SWIG_APPEND_VALUE(res);
     }

     /* In the FREEARG typemaps, get rid of the C vector.  
        (This can be overridden if you want to keep the C vector.) */

     %typemap(freearg) C_TYPE **PARALLEL_VECTOROUTPUT, 
		       C_TYPE **PARALLEL_LISTOUTPUT
     {
        if ((*$1)!=NULL) free(*$1);
     }

%enddef

%define TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_OUTPUT_WITH_EXPR(C_TYPE, SCM_TO_C_EXPR, C_TO_SCM_EXPR, SCM_TYPE)
  TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_WITH_EXPR(C_TYPE, SCM_TO_C_EXPR, SCM_TYPE)
  TYPEMAP_PARALLEL_LIST_VECTOR_OUTPUT_WITH_EXPR(C_TYPE, C_TO_SCM_EXPR, SCM_TYPE)
%enddef

%define TYPEMAP_PARALLEL_LIST_VECTOR_INPUT(C_TYPE, SCM_TO_C, SCM_TYPE)
  TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_WITH_EXPR
     (C_TYPE, SCM_TO_C(swig_scm_value), SCM_TYPE)
%enddef

%define TYPEMAP_PARALLEL_LIST_VECTOR_OUTPUT(C_TYPE, C_TO_SCM, SCM_TYPE)
  TYPEMAP_PARALLEL_LIST_VECTOR_OUTPUT_WITH_EXPR
     (C_TYPE, C_TO_SCM(swig_c_value), SCM_TYPE)
%enddef

%define TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_OUTPUT(C_TYPE, SCM_TO_C, C_TO_SCM, SCM_TYPE)
  TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_OUTPUT_WITH_EXPR
    (C_TYPE, SCM_TO_C(swig_scm_value), C_TO_SCM(swig_c_value), SCM_TYPE)
%enddef

/* We use the macro to define typemaps for some standard types. */

TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_OUTPUT(bool, scm_is_true, scm_from_bool, boolean);
TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_OUTPUT(char, SCM_CHAR, SCM_MAKE_CHAR, char);
TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_OUTPUT(unsigned char, SCM_CHAR, SCM_MAKE_CHAR, char);
TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_OUTPUT(int, scm_to_int, scm_from_long, integer);
TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_OUTPUT(short, scm_to_int, scm_from_long, integer);
TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_OUTPUT(long, scm_to_long, scm_from_long, integer);
TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_OUTPUT(ptrdiff_t, scm_to_long, scm_from_long, integer);
TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_OUTPUT(unsigned int, scm_to_ulong, scm_from_ulong, integer);
TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_OUTPUT(unsigned short, scm_to_ulong, scm_from_ulong, integer);
TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_OUTPUT(unsigned long, scm_to_ulong, scm_from_ulong, integer);
TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_OUTPUT(size_t, scm_to_ulong, scm_from_ulong, integer);
TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_OUTPUT(float, scm_to_double, scm_from_double, real);
TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_OUTPUT(double, scm_to_double, scm_from_double, real);
TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_OUTPUT(char *, SWIG_scm2str, SWIG_str02scm, string);
TYPEMAP_PARALLEL_LIST_VECTOR_INPUT_OUTPUT(const char *, SWIG_scm2str, SWIG_str02scm, string);

%typemap(freearg) char **PARALLEL_LISTINPUT, char **PARALLEL_VECTORINPUT,
    const char **PARALLEL_LISTINPUT, const char **PARALLEL_VECTORINPUT
{
    if (($1)!=NULL) {
	int i;
	for (i = 0; i<*_global_list_length; i++)
	    if (($1)[i] != NULL) SWIG_free(($1)[i]);
	SWIG_free($1);
    }
}

%typemap(freearg) char ***PARALLEL_LISTOUTPUT, char ***PARALLEL_VECTOROUTPUT,
    const char ***PARALLEL_LISTOUTPUT, const char ***PARALLEL_VECTOROUTPUT
{
    if ((*$1)!=NULL) {
	int i;
	for (i = 0; i<_global_arraylentemp; i++)
	    if ((*$1)[i] != NULL) free((*$1)[i]);
	free(*$1);
    }
}
