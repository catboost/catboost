/*
  struct timeval *
  time_t

  Ruby has builtin class Time.  INPUT/OUTPUT typemap for timeval and
  time_t is provided.

*/
%{
#ifdef __cplusplus
extern "C" {
#endif
#ifdef HAVE_SYS_TIME_H
# include <sys/time.h>
struct timeval rb_time_timeval(VALUE);
#endif
#ifdef __cplusplus
}
#endif
%}

%typemap(in) struct timeval *INPUT (struct timeval temp)
{
    if (NIL_P($input))
	$1 = NULL;
    else {
	temp = rb_time_timeval($input);
	$1 = &temp;
    }
}

%typemap(in,numinputs=0) struct timeval *OUTPUT(struct timeval temp)
{
    $1 = &temp;
}

%typemap(argout) struct timeval *OUTPUT
{
    $result = rb_time_new($1->tv_sec, $1->tv_usec);
}

%typemap(out) struct timeval *
{
    $result = rb_time_new($1->tv_sec, $1->tv_usec);
}

%typemap(out) struct timespec *
{
    $result = rb_time_new($1->tv_sec, $1->tv_nsec / 1000);
}

// time_t
%typemap(in) time_t
{
    if (NIL_P($input))
	$1 = (time_t)-1;
    else
	$1 = NUM2LONG(rb_funcall($input, rb_intern("tv_sec"), 0));
}

%typemap(typecheck) time_t
{
  $1 = (NIL_P($input) || TYPE(rb_funcall($input, rb_intern("respond_to?"), 1, ID2SYM(rb_intern("tv_sec")))) == T_TRUE);
}

%typemap(out) time_t
{
    $result = rb_time_new($1, 0);
}
