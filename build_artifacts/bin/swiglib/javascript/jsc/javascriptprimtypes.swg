/* ------------------------------------------------------------
 * Primitive Types
 * ------------------------------------------------------------ */

/* boolean */

%fragment(SWIG_From_frag(bool),"header") {
SWIGINTERNINLINE
JSValueRef SWIG_From_dec(bool)(bool value)
{
  return JSValueMakeBoolean(context, value);
}
}

%fragment(SWIG_AsVal_frag(bool),"header",
          fragment=SWIG_AsVal_frag(long)) {
SWIGINTERN
int SWIG_AsVal_dec(bool)(JSValueRef obj, bool *val)
{
  if(!JSValueIsBoolean(context, obj)) {
    return SWIG_ERROR;
  }
  if (val) *val = JSValueToBoolean(context, obj);
  return SWIG_OK;
}
}

/* int */

%fragment(SWIG_From_frag(int),"header") {
SWIGINTERNINLINE JSValueRef
  SWIG_From_dec(int)(int value)
{
  return JSValueMakeNumber(context, value);
}
}

/* long */

%fragment(SWIG_From_frag(long),"header") {
SWIGINTERNINLINE JSValueRef
SWIG_From_dec(long)(long value)
{
  return JSValueMakeNumber(context, value);
}
}

%fragment(SWIG_AsVal_frag(long),"header",
          fragment="SWIG_CanCastAsInteger") {
SWIGINTERN int
SWIG_AsVal_dec(long)(JSValueRef obj, long* val)
{
  if (!JSValueIsNumber(context, obj)) {
    return SWIG_TypeError;
  }
  if(val) *val = (long) JSValueToNumber(context, obj, NULL);

  return SWIG_OK;
}
}

/* unsigned long */

%fragment(SWIG_From_frag(unsigned long),"header",
          fragment=SWIG_From_frag(long)) {
SWIGINTERNINLINE JSValueRef
SWIG_From_dec(unsigned long)(unsigned long value)
{
  return (value > LONG_MAX) ?
    JSValueMakeNumber(context, value) : JSValueMakeNumber(context, %numeric_cast(value,long));
}
}

%fragment(SWIG_AsVal_frag(unsigned long),"header",
          fragment="SWIG_CanCastAsInteger") {
SWIGINTERN int
SWIG_AsVal_dec(unsigned long)(JSValueRef obj, unsigned long *val)
{
  long longVal;
  if(!JSValueIsNumber(context, obj)) {
    return SWIG_TypeError;
  }

  longVal = (long) JSValueToNumber(context, obj, NULL);

  if(longVal < 0) {
      return SWIG_OverflowError;
  }

  if(val) *val = longVal;

  return SWIG_OK;
}
}

/* long long */
// Note: these are copied from 'long' and probably need fixing

%fragment(SWIG_From_frag(long long),"header",
    fragment=SWIG_From_frag(long),
    fragment="SWIG_LongLongAvailable") {
%#ifdef SWIG_LONG_LONG_AVAILABLE
SWIGINTERNINLINE JSValueRef
SWIG_From_dec(long long)(long long value)
{
  return JSValueMakeNumber(context, value);
}
%#endif
}

%fragment(SWIG_AsVal_frag(long long),"header",
    fragment=SWIG_AsVal_frag(long),
    fragment="SWIG_CanCastAsInteger",
    fragment="SWIG_LongLongAvailable") {
%#ifdef SWIG_LONG_LONG_AVAILABLE
SWIGINTERN int
SWIG_AsVal_dec(long long)(JSValueRef obj, long long* val)
{
  if (!JSValueIsNumber(context, obj)) {
    return SWIG_TypeError;
  }
  if(val) *val = (long long) JSValueToNumber(context, obj, NULL);

  return SWIG_OK;
}
%#endif
}

/* unsigned long long */
// Note: these are copied from 'unsigned long' and probably need fixing

%fragment(SWIG_From_frag(unsigned long long),"header",
	  fragment=SWIG_From_frag(long long),
	  fragment="SWIG_LongLongAvailable") {
%#ifdef SWIG_LONG_LONG_AVAILABLE
SWIGINTERN JSValueRef
SWIG_From_dec(unsigned long long)(unsigned long long value)
{
  return (value > LONG_MAX) ?
    JSValueMakeNumber(context, value) : JSValueMakeNumber(context, %numeric_cast(value,long));
}
%#endif
}

%fragment(SWIG_AsVal_frag(unsigned long long),"header",
    fragment=SWIG_AsVal_frag(unsigned long),
    fragment="SWIG_CanCastAsInteger",
   fragment="SWIG_LongLongAvailable") {
%#ifdef SWIG_LONG_LONG_AVAILABLE
SWIGINTERN int
SWIG_AsVal_dec(unsigned long long)(JSValueRef obj, unsigned long long *val)
{
  long long longVal;
  if(!JSValueIsNumber(context, obj)) {
    return SWIG_TypeError;
  }

  longVal = (unsigned long long) JSValueToNumber(context, obj, NULL);

  if(longVal < 0) {
      return SWIG_OverflowError;
  }

  if(val) *val = longVal;

  return SWIG_OK;
}
%#endif
}

/* double */

%fragment(SWIG_From_frag(double),"header") {
SWIGINTERN JSValueRef
SWIG_From_dec(double) (double val)
{
  return JSValueMakeNumber(context, val);
}
}

%fragment(SWIG_AsVal_frag(double),"header") {
SWIGINTERN int
SWIG_AsVal_dec(double)(JSValueRef obj, double *val)
{
  if(!JSValueIsNumber(context, obj)) {
    return SWIG_TypeError;
  }
  if(val) *val = JSValueToNumber(context, obj, NULL);

  return SWIG_OK;
}
}
