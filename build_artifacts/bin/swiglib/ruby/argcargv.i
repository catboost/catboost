/* ------------------------------------------------------------
 * --- Argc & Argv ---
 * ------------------------------------------------------------ */
 
/* ------------------------------------------------------------

   Use it as follow:

     %apply (int ARGC, char **ARGV) { (size_t argc, const char **argv) }

     %inline %{

     int mainApp(size_t argc, const char **argv) 
     {
       return argc;
     }

   then in the ruby side:

     args = ["asdf", "asdf2"]
     mainApp(args);

 * ------------------------------------------------------------ */

%typemap(in) (int ARGC, char **ARGV) {
  if (rb_obj_is_kind_of($input,rb_cArray)) {
    int i;
    int size = RARRAY_LEN($input);
    $1 = ($1_ltype) size;
    $2 = (char **) malloc((size+1)*sizeof(char *));
    VALUE *ptr = RARRAY_PTR($input);
    for (i=0; i < size; i++, ptr++) {
      $2[i]= StringValuePtr(*ptr);
    }    
    $2[i]=NULL;
  } else {
    $1 = 0; $2 = 0;
    %argument_fail(SWIG_TypeError, "int ARGC, char **ARGV", $symname, $argnum);
  }
}

%typemap(typecheck, precedence=SWIG_TYPECHECK_STRING_ARRAY) (int ARGC, char **ARGV) {
  $1 = rb_obj_is_kind_of($input,rb_cArray);
}

%typemap(freearg) (int ARGC, char **ARGV) {
  free((char *) $2);
}
