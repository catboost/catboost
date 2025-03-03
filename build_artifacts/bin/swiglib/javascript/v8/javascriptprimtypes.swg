/* ------------------------------------------------------------
 * Primitive Types
 * ------------------------------------------------------------ */

/* boolean */

%fragment(SWIG_From_frag(bool),"header") {
SWIGINTERNINLINE
v8::Handle<v8::Value>
SWIG_From_dec(bool)(bool value)
{
  return SWIGV8_BOOLEAN_NEW(value);
}
}

%fragment(SWIG_AsVal_frag(bool),"header",
          fragment=SWIG_AsVal_frag(long)) {
SWIGINTERN
int SWIG_AsVal_dec(bool)(v8::Handle<v8::Value> obj, bool *val)
{
  if(!obj->IsBoolean()) {
    return SWIG_ERROR;
  }

  if (val) *val = SWIGV8_BOOLEAN_VALUE(obj);
  return SWIG_OK;
}
}

/* int */

%fragment(SWIG_From_frag(int),"header") {
SWIGINTERNINLINE
v8::Handle<v8::Value> SWIG_From_dec(int)(int value)
{
  return SWIGV8_INT32_NEW(value);
}
}

%fragment(SWIG_AsVal_frag(int),"header") {
SWIGINTERN
int SWIG_AsVal_dec(int)(v8::Handle<v8::Value> valRef, int* val)
{
  if (!valRef->IsNumber()) {
    return SWIG_TypeError;
  }
  if(val) *val = SWIGV8_INTEGER_VALUE(valRef);

  return SWIG_OK;
}
}

/* long */

%fragment(SWIG_From_frag(long),"header") {
SWIGINTERNINLINE
v8::Handle<v8::Value> SWIG_From_dec(long)(long value)
{
  return SWIGV8_NUMBER_NEW(value);
}
}

%fragment(SWIG_AsVal_frag(long),"header",
          fragment="SWIG_CanCastAsInteger") {
SWIGINTERN
int SWIG_AsVal_dec(long)(v8::Handle<v8::Value> obj, long* val)
{
  if (!obj->IsNumber()) {
    return SWIG_TypeError;
  }
  if(val) *val = (long) SWIGV8_INTEGER_VALUE(obj);

  return SWIG_OK;
}
}

/* unsigned long */

%fragment(SWIG_From_frag(unsigned long),"header",
          fragment=SWIG_From_frag(long)) {
SWIGINTERNINLINE
v8::Handle<v8::Value> SWIG_From_dec(unsigned long)(unsigned long value)
{
  return (value > LONG_MAX) ?
    SWIGV8_INTEGER_NEW_UNS(value) : SWIGV8_INTEGER_NEW(%numeric_cast(value,long));
}
}

%fragment(SWIG_AsVal_frag(unsigned long),"header",
          fragment="SWIG_CanCastAsInteger") {
SWIGINTERN
int SWIG_AsVal_dec(unsigned long)(v8::Handle<v8::Value> obj, unsigned long *val)
{
  if(!obj->IsNumber()) {
    return SWIG_TypeError;
  }

  long longVal = (long) SWIGV8_NUMBER_VALUE(obj);

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
SWIGINTERNINLINE
v8::Handle<v8::Value> SWIG_From_dec(long long)(long long value)
{
  return SWIGV8_NUMBER_NEW(value);
}
%#endif
}

%fragment(SWIG_AsVal_frag(long long),"header",
    fragment=SWIG_AsVal_frag(long),
    fragment="SWIG_CanCastAsInteger",
    fragment="SWIG_LongLongAvailable") {
%#ifdef SWIG_LONG_LONG_AVAILABLE
SWIGINTERN
int SWIG_AsVal_dec(long long)(v8::Handle<v8::Value> obj, long long* val)
{
  if (!obj->IsNumber()) {
    return SWIG_TypeError;
  }
  if(val) *val = (long long) SWIGV8_INTEGER_VALUE(obj);

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
SWIGINTERNINLINE
v8::Handle<v8::Value> SWIG_From_dec(unsigned long long)(unsigned long long value)
{
  return (value > LONG_MAX) ?
    SWIGV8_INTEGER_NEW_UNS(value) : SWIGV8_INTEGER_NEW(%numeric_cast(value,long));
}
%#endif
}

%fragment(SWIG_AsVal_frag(unsigned long long),"header",
    fragment=SWIG_AsVal_frag(unsigned long),
    fragment="SWIG_CanCastAsInteger",
    fragment="SWIG_LongLongAvailable") {
%#ifdef SWIG_LONG_LONG_AVAILABLE
SWIGINTERN
int SWIG_AsVal_dec(unsigned long long)(v8::Handle<v8::Value> obj, unsigned long long *val)
{
  if(!obj->IsNumber()) {
    return SWIG_TypeError;
  }

  long long longVal = (long long) SWIGV8_NUMBER_VALUE(obj);

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
SWIGINTERN
v8::Handle<v8::Value> SWIG_From_dec(double) (double val)
{
  return SWIGV8_NUMBER_NEW(val);
}
}

%fragment(SWIG_AsVal_frag(double),"header") {
SWIGINTERN
int SWIG_AsVal_dec(double)(v8::Handle<v8::Value> obj, double *val)
{
  if(!obj->IsNumber()) {
    return SWIG_TypeError;
  }
  if(val) *val = SWIGV8_NUMBER_VALUE(obj);

  return SWIG_OK;
}
}

