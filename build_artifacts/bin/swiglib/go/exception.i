%typemap(throws,noblock=1) (...) {
  SWIG_exception(SWIG_RuntimeError,"unknown exception");
}

%insert("runtime") %{
#define SWIG_exception(code, msg) _swig_gopanic(msg)
%}
