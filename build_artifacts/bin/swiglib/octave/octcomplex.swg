/*
  Defines the As/From conversors for double/float complex, you need to
  provide complex Type, the Name you want to use in the conversors,
  the complex Constructor method, and the Real and Imag complex
  accesor methods.

  See the std_complex.i and ccomplex.i for concrete examples.
*/

/* the common from conversor */
%define %swig_fromcplx_conv(Type, OctConstructor, Real, Imag)
     %fragment(SWIG_From_frag(Type),"header")
{
  SWIGINTERNINLINE octave_value
    SWIG_From(Type)(const Type& c)
    {
      return octave_value(OctConstructor(Real(c), Imag(c)));
    }
}
%enddef

// the double case
%define %swig_cplxdbl_conv(Type, Constructor, Real, Imag)
     %fragment(SWIG_AsVal_frag(Type),"header",
	       fragment=SWIG_AsVal_frag(double))
{
  SWIGINTERN int
    SWIG_AsVal(Type) (const octave_value& ov, Type* val)
    {
      if (ov.is_complex_scalar()) {
	if (val) {
	  Complex c(ov.complex_value());
	  *val=Constructor(c.real(),c.imag());
	}
	return SWIG_OK;
      } else {
	double d;    
	int res = SWIG_AddCast(SWIG_AsVal(double)(ov, &d));
	if (SWIG_IsOK(res)) {
	  if (val)
	    *val = Constructor(d, 0.0);
	  return res;
	}
      }
      return SWIG_TypeError;
    }
}
%swig_fromcplx_conv(Type, Complex, Real, Imag);
%enddef

// the float case
%define %swig_cplxflt_conv(Type, Constructor, Real, Imag)
     %fragment(SWIG_AsVal_frag(Type),"header",
	       fragment=SWIG_AsVal_frag(float)) {
  SWIGINTERN int
    SWIG_AsVal(Type) (const octave_value& ov, Type* val)
    {
      if (ov.is_complex_scalar()) {
	if (val) {
	  Complex c(ov.complex_value());
	  double re = c.real();
	  double im = c.imag();
	  if ((-FLT_MAX <= re && re <= FLT_MAX) && (-FLT_MAX <= im && im <= FLT_MAX)) {
	    if (val)
	      *val = Constructor(%numeric_cast(re, float),
				 %numeric_cast(im, float));
	    return SWIG_OK;
	  } else
	    return SWIG_OverflowError;
	}
      } else {
	float d;    
	int res = SWIG_AddCast(SWIG_AsVal(float)(ov, &d));
	if (SWIG_IsOK(res)) {
	  if (val)
	    *val = Constructor(d, 0.0);
	  return res;
	}
      }
      return SWIG_TypeError;
    }
}

%swig_fromcplx_conv(Type, FloatComplex, Real, Imag);
%enddef

#define %swig_cplxflt_convn(Type, Constructor, Real, Imag) \
%swig_cplxflt_conv(Type, Constructor, Real, Imag)


#define %swig_cplxdbl_convn(Type, Constructor, Real, Imag) \
%swig_cplxdbl_conv(Type, Constructor, Real, Imag)
