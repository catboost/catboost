/*
int PROG_ARGC
char **PROG_ARGV

    Some C function receive argc and argv from C main function.
    This typemap provides ignore typemap which pass Ruby ARGV contents
    as argc and argv to C function.
*/



// argc and argv
%typemap(in,numinputs=0) int PROG_ARGC {
    $1 = RARRAY_LEN(rb_argv) + 1;
}

%typemap(in,numinputs=0) char **PROG_ARGV {
    int i, n;
    VALUE ary = rb_eval_string("[$0] + ARGV");
    n = RARRAY_LEN(ary);
    $1 = (char **)malloc(n + 1);
    for (i = 0; i < n; i++) {
	VALUE v = rb_obj_as_string(RARRAY_PTR(ary)[i]);
	$1[i] = (char *)malloc(RSTRING_LEN(v) + 1);
	strcpy($1[i], RSTRING_PTR(v));
    }
}

%typemap(freearg) char **PROG_ARGV {
    int i, n = RARRAY_LEN(rb_argv) + 1;
    for (i = 0; i < n; i++) free($1[i]);
    free($1);
}

