// Pairs

%include <octstdcommon.swg>

//#define SWIG_STD_PAIR_ASVAL

%fragment("StdPairTraits","header",fragment="StdTraits") {
  namespace swig {
#ifdef SWIG_STD_PAIR_ASVAL
    template <class T, class U >
    struct traits_asval<std::pair<T,U> >  {
      typedef std::pair<T,U> value_type;

      static int get_pair(const octave_value& first, octave_value second,
			  std::pair<T,U> *val)
      {
	if (val) {
	  T *pfirst = &(val->first);
	  int res1 = swig::asval(first, pfirst);
	  if (!SWIG_IsOK(res1))
	    return res1;
	  U *psecond = &(val->second);
	  int res2 = swig::asval(second, psecond);
	  if (!SWIG_IsOK(res2))
	    return res2;
	  return res1 > res2 ? res1 : res2;
	} else {
	  T *pfirst = 0;
	  int res1 = swig::asval(first, pfirst);
	  if (!SWIG_IsOK(res1))
	    return res1;
	  U *psecond = 0;
	  int res2 = swig::asval((PyObject*)second, psecond);
	  if (!SWIG_IsOK(res2))
	    return res2;
	  return res1 > res2 ? res1 : res2;
	}
      }

      static int asval(const octave_value& obj, std::pair<T,U> *val) {
	if (
%#if SWIG_OCTAVE_PREREQ(4,4,0)
          obj.iscell()
%#else
          obj.is_cell()
%#endif
        ) {
	  Cell c=obj.cell_value();
	  if (c.numel()<2) {
	    error("pair from Cell array requires at least two elements");
	    return SWIG_ERROR;
	  }
	  return get_pair(c(0),c(1),val);
	} else {
	  value_type *p;
	  swig_type_info *descriptor = swig::type_info<value_type>();
	  int res = descriptor ? SWIG_ConvertPtr(obj, (void **)&p, descriptor, 0) : SWIG_ERROR;
	  if (SWIG_IsOK(res) && val)
	    *val = *p;
	  return res;
	}
	return SWIG_ERROR;
      }
    };

#else
    template <class T, class U >
    struct traits_asptr<std::pair<T,U> >  {
      typedef std::pair<T,U> value_type;

      static int get_pair(const octave_value& first, octave_value second,
			  std::pair<T,U> **val) 
      {
	if (val) {
	  value_type *vp = %new_instance(std::pair<T,U>);
	  T *pfirst = &(vp->first);
	  int res1 = swig::asval(first, pfirst);
	  if (!SWIG_IsOK(res1)) {
	    %delete(vp);
	    return res1;
	  }
	  U *psecond = &(vp->second);
	  int res2 = swig::asval(second, psecond);
	  if (!SWIG_IsOK(res2)) {
	    %delete(vp);
	    return res2;
	  }
	  *val = vp;
	  return SWIG_AddNewMask(res1 > res2 ? res1 : res2);
	} else {
	  T *pfirst = 0;
	  int res1 = swig::asval(first, pfirst);
	  if (!SWIG_IsOK(res1))
	    return res1;
	  U *psecond = 0;
	  int res2 = swig::asval(second, psecond);
	  if (!SWIG_IsOK(res2))
	    return res2;
	  return res1 > res2 ? res1 : res2;
	}
	return SWIG_ERROR;
      }

      static int asptr(const octave_value& obj, std::pair<T,U> **val) {
	if (
%#if SWIG_OCTAVE_PREREQ(4,4,0)
          obj.iscell()
%#else
          obj.is_cell()
%#endif
        ) {
	  Cell c=obj.cell_value();
	  if (c.numel()<2) {
	    error("pair from Cell array requires at least two elements");
	    return SWIG_ERROR;
	  }
	  return get_pair(c(0),c(1),val);
	} else {
	  value_type *p;
	  swig_type_info *descriptor = swig::type_info<value_type>();
	  int res = descriptor ? SWIG_ConvertPtr(obj, (void **)&p, descriptor, 0) : SWIG_ERROR;
	  if (SWIG_IsOK(res) && val)
	    *val = p;
	  return res;
	}
	return SWIG_ERROR;
      }
    };

#endif
    template <class T, class U >
    struct traits_from<std::pair<T,U> >   {
      static octave_value from(const std::pair<T,U>& val) {
	Cell c(1,2);
	c(0)=swig::from(val.first);
	c(1)=swig::from(val.second);
	return c;
      }
    };
  }
}

%define %swig_pair_methods(pair...)
%enddef

%include <std/std_pair.i>

