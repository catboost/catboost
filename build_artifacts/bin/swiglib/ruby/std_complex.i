/*
 *  STD C++ complex typemaps
 */

%include <rubycomplex.swg>

%{
#include <complex> 
%}

namespace std {
  %naturalvar complex;
  template<typename T> class complex;
  %template() complex<double>;
  %template() complex<float>;
}

/* defining the complex as/from converters */

%swig_cplxdbl_convn(std::complex<double>, std::complex<double>, std::real, std::imag)
%swig_cplxflt_convn(std::complex<float>,  std::complex<float>,  std::real, std::imag)

/* defining the typemaps */

%typemaps_primitive(%checkcode(CPLXDBL), std::complex<double>);
%typemaps_primitive(%checkcode(CPLXFLT), std::complex<float>);



