/* ------------------------------------------------------------
 * Primitive Types
 * ------------------------------------------------------------ */

/* bool */

%fragment(SWIG_From_frag(bool),"header") {
SWIGINTERNINLINE SV *
SWIG_From_dec(bool)(bool value)
{
  return boolSV(value);
}
}

%fragment(SWIG_AsVal_frag(bool),"header") {
SWIGINTERN int
SWIG_AsVal_dec(bool)(SV *obj, bool* val)
{
  if (obj == &PL_sv_yes) {
    if (val) *val = true;
    return SWIG_OK;
  } else if (obj == &PL_sv_no) { 
    if (val) *val = false;
    return SWIG_OK;
  } else {
    if (val) *val = SvTRUE(obj) ? true : false;
    return SWIG_AddCast(SWIG_OK);    
  }
}
}


/* long */

%fragment(SWIG_From_frag(long),"header") {
SWIGINTERNINLINE SV *
SWIG_From_dec(long)(long value)
{
  SV *sv;
  if (IVSIZE >= sizeof(value) || (value >= IV_MIN && value <= IV_MAX))
    sv = newSViv(value);
  else
    sv = newSVpvf("%ld", value);
  return sv_2mortal(sv);
}
}

%fragment(SWIG_AsVal_frag(long),"header",
	  fragment="<limits.h>",
	  fragment="<stdlib.h>",
	  fragment="SWIG_CanCastAsInteger") {
SWIGINTERN int
SWIG_AsVal_dec(long)(SV *obj, long* val)
{
  if (SvUOK(obj)) {
    UV v = SvUV(obj);
    if (UVSIZE < sizeof(*val) || v <= LONG_MAX) {
      if (val) *val = v;
      return SWIG_OK;
    }
    return SWIG_OverflowError;
  } else if (SvIOK(obj)) {
    IV v = SvIV(obj);
    if (IVSIZE <= sizeof(*val) || (v >= LONG_MIN && v <= LONG_MAX)) {
      if(val) *val = v;
      return SWIG_OK;
    }
    return SWIG_OverflowError;
  } else {
    int dispatch = 0;
    const char *nptr = SvPV_nolen(obj);
    if (nptr) {
      char *endptr;
      long v;
      errno = 0;
      v = strtol(nptr, &endptr,0);
      if (errno == ERANGE) {
	errno = 0;
	return SWIG_OverflowError;
      } else {
	if (*endptr == '\0') {
	  if (val) *val = v;
	  return SWIG_Str2NumCast(SWIG_OK);
	}
      }
    }
    if (!dispatch) {
      double d;
      int res = SWIG_AddCast(SWIG_AsVal(double)(obj,&d));
      if (SWIG_IsOK(res) && SWIG_CanCastAsInteger(&d, LONG_MIN, LONG_MAX)) {
	if (val) *val = (long)(d);
	return res;
      }
    }
  }
  return SWIG_TypeError;
}
}

/* unsigned long */

%fragment(SWIG_From_frag(unsigned long),"header") {
SWIGINTERNINLINE SV *
SWIG_From_dec(unsigned long)(unsigned long value)
{
  SV *sv;
  if (UVSIZE >= sizeof(value) || value <= UV_MAX)
    sv = newSVuv(value);
  else
    sv = newSVpvf("%lu", value);
  return sv_2mortal(sv);
}
}

%fragment(SWIG_AsVal_frag(unsigned long),"header",
	  fragment="<limits.h>",
	  fragment="<stdlib.h>",
	  fragment="SWIG_CanCastAsInteger") {
SWIGINTERN int
SWIG_AsVal_dec(unsigned long)(SV *obj, unsigned long *val) 
{
  if (SvUOK(obj)) {
    UV v = SvUV(obj);
    if (UVSIZE <= sizeof(*val) || v <= ULONG_MAX) {
      if (val) *val = v;
      return SWIG_OK;
    }
    return SWIG_OverflowError;
  } else if (SvIOK(obj)) {
    IV v = SvIV(obj);
    if (v >= 0 && (IVSIZE <= sizeof(*val) || v <= ULONG_MAX)) {
      if (val) *val = v;
      return SWIG_OK;
    }
    return SWIG_OverflowError;
  } else {
    int dispatch = 0;
    const char *nptr = SvPV_nolen(obj);
    if (nptr) {
      char *endptr;
      unsigned long v;
      errno = 0;
      v = strtoul(nptr, &endptr,0);
      if (errno == ERANGE) {
	errno = 0;
	return SWIG_OverflowError;
      } else {
	if (*endptr == '\0') {
	  if (val) *val = v;
	  return SWIG_Str2NumCast(SWIG_OK);
	}
      }
    }
    if (!dispatch) {
      double d;
      int res = SWIG_AddCast(SWIG_AsVal(double)(obj,&d));
      if (SWIG_IsOK(res) && SWIG_CanCastAsInteger(&d, 0, ULONG_MAX)) {
	if (val) *val = (unsigned long)(d);
	return res;
      }
    }
  }
  return SWIG_TypeError;
}
}

/* long long */

%fragment(SWIG_From_frag(long long),"header",
	  fragment="SWIG_LongLongAvailable",
	  fragment="<stdio.h>") {
%#ifdef SWIG_LONG_LONG_AVAILABLE
SWIGINTERNINLINE SV *
SWIG_From_dec(long long)(long long value)
{
  SV *sv;
  if (IVSIZE >= sizeof(value) || (value >= IV_MIN && value <= IV_MAX))
    sv = newSViv((IV)(value));
  else {
    //sv = newSVpvf("%lld", value); doesn't work in non 64bit Perl
    char temp[256];
    SWIG_snprintf(temp, sizeof(temp), "%lld", value);
    sv = newSVpv(temp, 0);
  }
  return sv_2mortal(sv);
}
%#endif
}

%fragment(SWIG_AsVal_frag(long long),"header",
	  fragment="SWIG_LongLongAvailable",
	  fragment="<stdlib.h>",
	  fragment="SWIG_CanCastAsInteger") {
%#ifdef SWIG_LONG_LONG_AVAILABLE
SWIGINTERN int
SWIG_AsVal_dec(long long)(SV *obj, long long *val)
{
  if (SvUOK(obj)) {
    UV v = SvUV(obj);
    /* pretty sure this could allow v == LLONG MAX */
    if (UVSIZE < sizeof(*val) || v < LLONG_MAX) {
      if (val) *val = v;
      return SWIG_OK;
    }
    return SWIG_OverflowError;
  } else if (SvIOK(obj)) {
    IV v = SvIV(obj);
    if (IVSIZE <= sizeof(*val) || (v >= LLONG_MIN && v <= LLONG_MAX)) {
      if (val) *val = v;
      return SWIG_OK;
    }
    return SWIG_OverflowError;
  } else {
    int dispatch = 0;
    const char *nptr = SvPV_nolen(obj);
    if (nptr) {
      char *endptr;
      long long v;
      errno = 0;
      v = strtoll(nptr, &endptr,0);
      if (errno == ERANGE) {
	errno = 0;
	return SWIG_OverflowError;
      } else {
	if (*endptr == '\0') {
	  if (val) *val = v;
	  return SWIG_Str2NumCast(SWIG_OK);
	}
      }
    }
    if (!dispatch) {
      const double mant_max = 1LL << DBL_MANT_DIG;
      const double mant_min = -mant_max;
      double d;
      int res = SWIG_AddCast(SWIG_AsVal(double)(obj,&d));
      if (SWIG_IsOK(res) && SWIG_CanCastAsInteger(&d, mant_min, mant_max)) {
	if (val) *val = (long long)(d);
	return res;
      }
    }
  }
  return SWIG_TypeError; 
}
%#endif
}

/* unsigned long long */

%fragment(SWIG_From_frag(unsigned long long),"header",
	  fragment="SWIG_LongLongAvailable",
	  fragment="<stdio.h>") {
%#ifdef SWIG_LONG_LONG_AVAILABLE
SWIGINTERNINLINE SV *
SWIG_From_dec(unsigned long long)(unsigned long long value)
{
  SV *sv;
  if (UVSIZE >= sizeof(value) || value <= UV_MAX)
    sv = newSVuv((UV)(value));
  else {
    //sv = newSVpvf("%llu", value); doesn't work in non 64bit Perl
    char temp[256];
    SWIG_snprintf(temp, sizeof(temp), "%llu", value);
    sv = newSVpv(temp, 0);
  }
  return sv_2mortal(sv);
}
%#endif
}

%fragment(SWIG_AsVal_frag(unsigned long long),"header",
	  fragment="SWIG_LongLongAvailable",
	  fragment="<stdlib.h>",
	  fragment="SWIG_CanCastAsInteger") {
%#ifdef SWIG_LONG_LONG_AVAILABLE
SWIGINTERN int
SWIG_AsVal_dec(unsigned long long)(SV *obj, unsigned long long *val)
{
  if (SvUOK(obj)) {
    /* pretty sure this should be conditional on
     * (UVSIZE <= sizeof(*val) || v <= ULLONG_MAX) */
    if (val) *val = SvUV(obj);
    return SWIG_OK;
  } else if (SvIOK(obj)) {
    IV v = SvIV(obj);
    if (v >= 0 && (IVSIZE <= sizeof(*val) || v <= ULLONG_MAX)) {
      if (val) *val = v;
      return SWIG_OK;
    } else {
      return SWIG_OverflowError;
    }
  } else {
    int dispatch = 0;
    const char *nptr = SvPV_nolen(obj);
    if (nptr) {
      char *endptr;
      unsigned long long v;
      errno = 0;
      v = strtoull(nptr, &endptr,0);
      if (errno == ERANGE) {
	errno = 0;
	return SWIG_OverflowError;
      } else {
	if (*endptr == '\0') {
	  if (val) *val = v;
	  return SWIG_Str2NumCast(SWIG_OK);
	}
      }
    }
    if (!dispatch) {
      const double mant_max = 1LL << DBL_MANT_DIG;
      double d;
      int res = SWIG_AddCast(SWIG_AsVal(double)(obj,&d));
      if (SWIG_IsOK(res) && SWIG_CanCastAsInteger(&d, 0, mant_max)) {
	if (val) *val = (unsigned long long)(d);
	return res;
      }
    }
  }
  return SWIG_TypeError;
}
%#endif
}

/* double */

%fragment(SWIG_From_frag(double),"header") {
SWIGINTERNINLINE SV *
SWIG_From_dec(double)(double value)
{
  return sv_2mortal(newSVnv(value));
}
}

%fragment(SWIG_AsVal_frag(double),"header") {
SWIGINTERN int
SWIG_AsVal_dec(double)(SV *obj, double *val)
{
  if (SvNIOK(obj)) {
    if (val) *val = SvNV(obj);
    return SWIG_OK;
  } else if (SvIOK(obj)) {
    if (val) *val = (double) SvIV(obj);
    return SWIG_AddCast(SWIG_OK);
  } else {
    const char *nptr = SvPV_nolen(obj);
    if (nptr) {
      char *endptr;
      double v;
      errno = 0;
      v = strtod(nptr, &endptr);
      if (errno == ERANGE) {
	errno = 0;
	return SWIG_OverflowError;
      } else {
	if (*endptr == '\0') {
	  if (val) *val = v;
	  return SWIG_Str2NumCast(SWIG_OK);
	}
      }
    }
  }
  return SWIG_TypeError;
}
}
