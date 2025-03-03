%include <typemaps/exception.swg>


%insert("runtime") {
  %define_as(SWIG_exception(code, msg), 
%block(switch (code) {case SWIG_IndexError: return Rf_ScalarLogical(NA_LOGICAL); default: %error(code, msg); SWIG_fail;} ))
}

