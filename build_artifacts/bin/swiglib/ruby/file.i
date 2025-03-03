// FILE *
%{
#ifdef __cplusplus
extern "C" {
#endif

/* Ruby 1.9 changed the file name of this header */
#ifdef HAVE_RUBY_IO_H
#include "ruby/io.h"
#else
#include "rubyio.h"
#endif

#ifdef __cplusplus
}
#endif
%}

%typemap(in) FILE *READ {
    OpenFile *of;
    GetOpenFile($input, of);
    rb_io_check_readable(of);
    $1 = GetReadFile(of);
    rb_read_check($1);
}

%typemap(in) FILE *READ_NOCHECK {
    OpenFile *of;
    GetOpenFile($input, of);
    rb_io_check_readable(of);
    $1 = GetReadFile(of);
}

%typemap(in) FILE *WRITE {
    OpenFile *of;
    GetOpenFile($input, of);
    rb_io_check_writable(of);
    $1 = GetWriteFile(of);
}
