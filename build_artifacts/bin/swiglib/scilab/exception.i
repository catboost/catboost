%include <typemaps/exception.swg>


%insert("runtime") {
  %define_as(SWIG_exception(code, msg), SWIG_Scilab_Error(code, msg);)
}
