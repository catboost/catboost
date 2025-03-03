/*
  Defines the As/From converters for double/float complex, you need to
  provide complex Type, the Name you want to use in the converters,
  the complex Constructor method, and the Real and Imag complex
  accessor methods.

  See the std_complex.i and ccomplex.i for concrete examples.
*/

/* the common from converter */
%define %swig_fromcplx_conv(Type, Real, Imag)
%fragment(SWIG_From_frag(Type),"header",
          fragment=SWIG_From_frag(double))
{
SWIGINTERNINLINE v8::Handle<v8::Value>
SWIG_From_dec(Type)(%ifcplusplus(const Type&, Type) c)
{
  SWIGV8_HANDLESCOPE_ESC();

  v8::Local<v8::Array> vals = SWIGV8_ARRAY_NEW();

  vals->Set(0, SWIG_From(double)(Real(c)));
  vals->Set(1, SWIG_From(double)(Imag(c)));
  SWIGV8_ESCAPE(vals);
}
}
%enddef

/* the double case */
%define %swig_cplxdbl_conv(Type, Constructor, Real, Imag)
%fragment(SWIG_AsVal_frag(Type),"header",
	  fragment=SWIG_AsVal_frag(double))
{
SWIGINTERN int
SWIG_AsVal_dec(Type) (v8::Handle<v8::Value> o, Type* val)
{
  SWIGV8_HANDLESCOPE();
  
  if (o->IsArray()) {
    v8::Handle<v8::Array> array = v8::Handle<v8::Array>::Cast(o);
    
    if(array->Length() != 2) SWIG_Error(SWIG_TypeError, "Illegal argument for complex: must be array[2].");
    double re, im;
    int res;

    res = SWIG_AsVal(double)(array->Get(0), &re);
    if(!SWIG_IsOK(res)) {
      return SWIG_TypeError;
    }
    
    res = SWIG_AsVal(double)(array->Get(1), &im);
    if(!SWIG_IsOK(res)) {
      return SWIG_TypeError;
    }
    
    if (val) *val = Constructor(re, im);
    return SWIG_OK;
  } else if(o->IsNumber()){
    double d;
    int res = SWIG_AddCast(SWIG_AsVal(double)(o, &d));
    if (SWIG_IsOK(res)) {
      if (val) *val = Constructor(d, 0.0);
      return res;
    }
  }
  return SWIG_TypeError;
}
}
%swig_fromcplx_conv(Type, Real, Imag);
%enddef

/* the float case */
%define %swig_cplxflt_conv(Type, Constructor, Real, Imag)
%fragment(SWIG_AsVal_frag(Type),"header",
          fragment=SWIG_AsVal_frag(float)) {
SWIGINTERN int
SWIG_AsVal_dec(Type) (v8::Handle<v8::Value> o, Type* val)
{
  SWIGV8_HANDLESCOPE();

  if (o->IsArray()) {
    v8::Handle<v8::Array> array = v8::Handle<v8::Array>::Cast(o);
    
    if(array->Length() != 2) SWIG_Error(SWIG_TypeError, "Illegal argument for complex: must be array[2].");
    double re, im;
    int res;

    res = SWIG_AsVal(double)(array->Get(0), &re);
    if(!SWIG_IsOK(res)) {
      return SWIG_TypeError;
    }
    
    res = SWIG_AsVal(double)(array->Get(1), &im);
    if(!SWIG_IsOK(res)) {
      return SWIG_TypeError;
    }
    
    if ((-FLT_MAX <= re && re <= FLT_MAX) && (-FLT_MAX <= im && im <= FLT_MAX)) {
      if (val) *val = Constructor(%numeric_cast(re, float),
                                  %numeric_cast(im, float));
      return SWIG_OK;
    } else {
      return SWIG_OverflowError;
    }    
  } else if(o->IsNumber()){
    float re;
    int res = SWIG_AddCast(SWIG_AsVal(float)(o, &re));
    if (SWIG_IsOK(res)) {
      if (val) *val = Constructor(re, 0.0);
      return res;
    }
  }
  return SWIG_TypeError;
}
}
%swig_fromcplx_conv(Type, Real, Imag);
%enddef

#define %swig_cplxflt_convn(Type, Constructor, Real, Imag) \
%swig_cplxflt_conv(Type, Constructor, Real, Imag)


#define %swig_cplxdbl_convn(Type, Constructor, Real, Imag) \
%swig_cplxdbl_conv(Type, Constructor, Real, Imag)
