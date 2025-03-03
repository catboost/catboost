/* ------------------------------------------------------------
 * Primitive Types
 * ------------------------------------------------------------ */


// boolean

%fragment(SWIG_From_frag(bool),"header") {
SWIGINTERNINLINE octave_value
  SWIG_From_dec(bool)(bool value)
{
  return octave_value(value);
}
}

%fragment(SWIG_AsVal_frag(bool),"header",
	  fragment=SWIG_AsVal_frag(long)) {
SWIGINTERN int
SWIG_AsVal_dec(bool)(const octave_value& ov, bool *val)
{
%#if SWIG_OCTAVE_PREREQ(4,4,0)
  if (!ov.islogical())
%#else
  if (!ov.is_bool_type())
%#endif
    return SWIG_ERROR;
  if (val)
    *val = ov.bool_value();
  return SWIG_OK;
}
}

// long

%fragment(SWIG_From_frag(long),"header") {
  SWIGINTERNINLINE octave_value SWIG_From_dec(long)  (long value)
    {    
      return octave_value(value);
    }
}


%fragment(SWIG_AsVal_frag(long),"header") {
  SWIGINTERN int SWIG_AsVal_dec(long)(const octave_value& ov, long* val)
    {
      if (!ov.is_scalar_type())
	return SWIG_TypeError;
      if (ov.is_complex_scalar())
	return SWIG_TypeError;
      if (ov.is_double_type()||ov.is_single_type()) {
	double v=ov.double_value();
	if (v!=floor(v))
	  return SWIG_TypeError;
      }
      if (val)
	*val = ov.long_value();
      return SWIG_OK;
    }
}

// unsigned long

%fragment(SWIG_From_frag(unsigned long),"header") {
  SWIGINTERNINLINE octave_value SWIG_From_dec(unsigned long)  (unsigned long value)
    {    
      return octave_value(value);
    }
}


%fragment(SWIG_AsVal_frag(unsigned long),"header") {
  SWIGINTERN int SWIG_AsVal_dec(unsigned long)(const octave_value& ov, unsigned long* val)
    {
      if (!ov.is_scalar_type())
	return SWIG_TypeError;
      if (ov.is_complex_scalar())
	return SWIG_TypeError;
      if (ov.is_double_type()||ov.is_single_type()) {
	double v=ov.double_value();
	if (v<0)
	  return SWIG_OverflowError;  
	if (v!=floor(v))
	  return SWIG_TypeError;
      }
      if (ov.is_int8_type()||ov.is_int16_type()||
	  ov.is_int32_type()) {
	long v=ov.long_value();
	if (v<0)
	  return SWIG_OverflowError;  
      }
      if (ov.is_int64_type()) {
	long long v=ov.int64_scalar_value().value();
	if (v<0)
	  return SWIG_OverflowError;  
      }
      if (val)
	*val = ov.ulong_value();
      return SWIG_OK;
    }
}

// long long

%fragment(SWIG_From_frag(long long),"header",
	  fragment="SWIG_LongLongAvailable") {
%#ifdef SWIG_LONG_LONG_AVAILABLE
  SWIGINTERNINLINE octave_value SWIG_From_dec(long long)  (long long value)
    {    
      return octave_int64(value);
    }
%#endif
}


%fragment(SWIG_AsVal_frag(long long),"header",
	  fragment="SWIG_LongLongAvailable") {
%#ifdef SWIG_LONG_LONG_AVAILABLE
  SWIGINTERN int SWIG_AsVal_dec(long long)(const octave_value& ov, long long* val)
    {
      if (!ov.is_scalar_type())
	return SWIG_TypeError;
      if (ov.is_complex_scalar())
	return SWIG_TypeError;
      if (ov.is_double_type()||ov.is_single_type()) {
	double v=ov.double_value();
	if (v!=floor(v))
	  return SWIG_TypeError;
      }
      if (val) {
	if (ov.is_int64_type())
	  *val = ov.int64_scalar_value().value();
	else if (ov.is_uint64_type())
	  *val = ov.uint64_scalar_value().value();
	else
	  *val = ov.long_value();
      }
      return SWIG_OK;
    }
%#endif
}

%fragment(SWIG_From_frag(unsigned long long),"header",
	  fragment="SWIG_LongLongAvailable") {
%#ifdef SWIG_LONG_LONG_AVAILABLE
  SWIGINTERNINLINE octave_value SWIG_From_dec(unsigned long long)  (unsigned long long value)
    {    
      return octave_uint64(value);
    }
%#endif
}

%fragment(SWIG_AsVal_frag(unsigned long long),"header",
	  fragment="SWIG_LongLongAvailable") {
%#ifdef SWIG_LONG_LONG_AVAILABLE
  SWIGINTERN int SWIG_AsVal_dec(unsigned long long)(const octave_value& ov, unsigned long long* val)
    {
      if (!ov.is_scalar_type())
	return SWIG_TypeError;
      if (ov.is_complex_scalar())
	return SWIG_TypeError;
      if (ov.is_double_type()||ov.is_single_type()) {
	double v=ov.double_value();
	if (v<0)
	  return SWIG_OverflowError;  
	if (v!=floor(v))
	  return SWIG_TypeError;
      }
      if (ov.is_int8_type()||ov.is_int16_type()||
	  ov.is_int32_type()) {
	long v=ov.long_value();
	if (v<0)
	  return SWIG_OverflowError;  
      }
      if (ov.is_int64_type()) {
	long long v=ov.int64_scalar_value().value();
	if (v<0)
	  return SWIG_OverflowError;  
      }
      if (val) {
	if (ov.is_int64_type())
	  *val = ov.int64_scalar_value().value();
	else if (ov.is_uint64_type())
	  *val = ov.uint64_scalar_value().value();
	else
	  *val = ov.long_value();
      }
      return SWIG_OK;
    }
%#endif
}

// double

%fragment(SWIG_From_frag(double),"header") {
  SWIGINTERNINLINE octave_value SWIG_From_dec(double)  (double value)
    {    
      return octave_value(value);
    }
}


%fragment(SWIG_AsVal_frag(double),"header") {
  SWIGINTERN int SWIG_AsVal_dec(double)(const octave_value& ov, double* val)
    {
      if (!ov.is_scalar_type())
	return SWIG_TypeError;
      if (ov.is_complex_scalar())
	return SWIG_TypeError;
      if (val)
	*val = ov.double_value();
      return SWIG_OK;
    }
}

// const char* (strings)

%fragment("SWIG_AsCharPtrAndSize","header") {
SWIGINTERN int
SWIG_AsCharPtrAndSize(octave_value ov, char** cptr, size_t* psize, int *alloc)
{
  if (
%#if SWIG_OCTAVE_PREREQ(4,4,0)
    ov.iscell()
%#else
    ov.is_cell()
%#endif
    && ov.rows() == 1 && ov.columns() == 1)
    ov = ov.cell_value()(0);
  if (!ov.is_string())
    return SWIG_TypeError;
  
  std::string str=ov.string_value();
  size_t len=str.size();
  char* cstr=(char*)str.c_str();
  if (alloc) {
    *cptr = %new_copy_array(cstr, len + 1, char);
    *alloc = SWIG_NEWOBJ;
  } else if (cptr)
    *cptr = cstr;
  if (psize)
    *psize = len + 1;
  return SWIG_OK;
}
}

%fragment("SWIG_FromCharPtrAndSize","header",fragment="SWIG_pchar_descriptor") {
SWIGINTERNINLINE octave_value
SWIG_FromCharPtrAndSize(const char* carray, size_t size)
{
  return std::string(carray,carray+size);
}
}


