/* -----------------------------------------------------------------------------
 * std_complex.i
 *
 * Typemaps for handling std::complex<float> and std::complex<double> as a .NET
 * System.Numerics.Complex type. Requires .NET 4 minimum.
 * ----------------------------------------------------------------------------- */

%{
#include <complex>
%}

%fragment("SwigSystemNumericsComplex", "header") {
extern "C" {
// Identical to the layout of System.Numerics.Complex, but does assume that it is
// LayoutKind.Sequential on the managed side
struct SwigSystemNumericsComplex {
  double real;
  double imag;
};
}

SWIGINTERN SwigSystemNumericsComplex SwigCreateSystemNumericsComplex(double real, double imag) {
  SwigSystemNumericsComplex cpx;
  cpx.real = real;
  cpx.imag = imag;
  return cpx;
}
}

namespace std {

%naturalvar complex;

template<typename T>
class complex
{
public:
    complex(T re = T(), T im = T());
};

}

%define SWIG_COMPLEX_TYPEMAPS(T)
%typemap(ctype, fragment="SwigSystemNumericsComplex") std::complex<T>, const std::complex<T> & "SwigSystemNumericsComplex"
%typemap(imtype) std::complex<T>, const std::complex<T> & "System.Numerics.Complex"
%typemap(cstype) std::complex<T>, const std::complex<T> & "System.Numerics.Complex"

%typemap(in) std::complex<T>
%{$1 = std::complex< double >($input.real, $input.imag);%}

%typemap(in) const std::complex<T> &($*1_ltype temp)
%{temp = std::complex< T >((T)$input.real, (T)$input.imag);
  $1 = &temp;%}

%typemap(out, null="SwigCreateSystemNumericsComplex(0.0, 0.0)") std::complex<T>
%{$result = SwigCreateSystemNumericsComplex($1.real(), $1.imag());%}

%typemap(out, null="SwigCreateSystemNumericsComplex(0.0, 0.0)") const std::complex<T> &
%{$result = SwigCreateSystemNumericsComplex($1->real(), $1->imag());%}

%typemap(cstype) std::complex<T>, const std::complex<T> & "System.Numerics.Complex"

%typemap(csin) std::complex<T>, const std::complex<T> & "$csinput"

%typemap(csout, excode=SWIGEXCODE) std::complex<T>, const std::complex<T> & {
    System.Numerics.Complex ret = $imcall;$excode
    return ret;
  }

%typemap(csvarin, excode=SWIGEXCODE2) const std::complex<T> & %{
    set {
      $imcall;$excode
    }
  %}

%typemap(csvarout, excode=SWIGEXCODE2) const std::complex<T> & %{
    get {
      System.Numerics.Complex ret = $imcall;$excode
      return ret;
    }
  %}

%template() std::complex<T>;
%enddef

// By default, typemaps for both std::complex<double> and std::complex<float>
// are defined, but one of them can be disabled by predefining the
// corresponding symbol before including this file.
#ifndef SWIG_NO_STD_COMPLEX_DOUBLE
SWIG_COMPLEX_TYPEMAPS(double)
#endif

#ifndef SWIG_NO_STD_COMPLEX_FLOAT
SWIG_COMPLEX_TYPEMAPS(float)
#endif
