// Maps

%include <octcontainer.swg>

%fragment("StdMapCommonTraits","header",fragment="StdSequenceTraits")
{
  namespace swig {
    template <class ValueType>
    struct from_key_oper 
    {
      typedef const ValueType& argument_type;
      typedef octave_value result_type;
      result_type operator()(argument_type v) const
      {
	return swig::from(v.first);
      }
    };

    template <class ValueType>
    struct from_value_oper 
    {
      typedef const ValueType& argument_type;
      typedef octave_value result_type;
      result_type operator()(argument_type v) const
      {
	return swig::from(v.second);
      }
    };

    template<class OutIterator, class FromOper, class ValueType = typename OutIterator::value_type>
    struct OctMapIterator_T : OctSwigIteratorClosed_T<OutIterator, ValueType, FromOper>
    {
      OctMapIterator_T(OutIterator curr, OutIterator first, OutIterator last, octave_value seq)
	: OctSwigIteratorClosed_T<OutIterator,ValueType,FromOper>(curr, first, last, seq)
      {
      }
    };


    template<class OutIterator,
	     class FromOper = from_key_oper<typename OutIterator::value_type> >
    struct OctMapKeyIterator_T : OctMapIterator_T<OutIterator, FromOper>
    {
      OctMapKeyIterator_T(OutIterator curr, OutIterator first, OutIterator last, octave_value seq)
	: OctMapIterator_T<OutIterator, FromOper>(curr, first, last, seq)
      {
      }
    };

    template<typename OutIter>
    inline OctSwigIterator*
    make_output_key_iterator(const OutIter& current, const OutIter& begin, const OutIter& end, octave_value seq = octave_value())
    {
      return new OctMapKeyIterator_T<OutIter>(current, begin, end, seq);
    }

    template<class OutIterator,
	     class FromOper = from_value_oper<typename OutIterator::value_type> >
    struct OctMapValueIterator_T : OctMapIterator_T<OutIterator, FromOper>
    {
      OctMapValueIterator_T(OutIterator curr, OutIterator first, OutIterator last, octave_value seq)
	: OctMapIterator_T<OutIterator, FromOper>(curr, first, last, seq)
      {
      }
    };
    

    template<typename OutIter>
    inline OctSwigIterator*
    make_output_value_iterator(const OutIter& current, const OutIter& begin, const OutIter& end, octave_value seq = 0)
    {
      return new OctMapValueIterator_T<OutIter>(current, begin, end, seq);
    }
  }
}

%fragment("StdMapTraits","header",fragment="StdMapCommonTraits")
{
  namespace swig {
    template <class OctSeq, class K, class T >
    inline void
    assign(const OctSeq& octseq, std::map<K,T > *map) {
      typedef typename std::map<K,T>::value_type value_type;
      typename OctSeq::const_iterator it = octseq.begin();
      for (;it != octseq.end(); ++it) {
	map->insert(value_type(it->first, it->second));
      }
    }

    template <class K, class T>
    struct traits_asptr<std::map<K,T> >  {
      typedef std::map<K,T> map_type;
      static int asptr(octave_value obj, map_type **val) {
	/*
	int res = SWIG_ERROR;
	if (PyDict_Check(obj)) {
	  SwigVar_PyObject items = PyObject_CallMethod(obj,(char *)"items",NULL);
	  res = traits_asptr_stdseq<std::map<K,T>, std::pair<K, T> >::asptr(items, val);
	} else {
	  map_type *p;
	  swig_type_info *descriptor = swig::type_info<map_type>();
	  res = descriptor ? SWIG_ConvertPtr(obj, (void **)&p, descriptor, 0) : SWIG_ERROR;
	  if (SWIG_IsOK(res) && val)  *val = p;
	}
	return res;
	*/
	return SWIG_ERROR;
      }      
    };
      
    template <class K, class T >
    struct traits_from<std::map<K,T> >  {
      typedef std::map<K,T> map_type;
      typedef typename map_type::const_iterator const_iterator;
      typedef typename map_type::size_type size_type;
            
      static octave_value from(const map_type& map) {
	/*
	swig_type_info *desc = swig::type_info<map_type>();
	if (desc && desc->clientdata) {
	  return SWIG_NewPointerObj(new map_type(map), desc, SWIG_POINTER_OWN);
	} else {
	  size_type size = map.size();
	  int pysize = (size <= (size_type) INT_MAX) ? (int) size : -1;
	  if (pysize < 0) {
	    SWIG_PYTHON_THREAD_BEGIN_BLOCK;
	    PyErr_SetString(PyExc_OverflowError,
			    "map size not valid in python");
	    SWIG_PYTHON_THREAD_END_BLOCK;
	    return NULL;
	  }
	  PyObject *obj = PyDict_New();
	  for (const_iterator i= map.begin(); i!= map.end(); ++i) {
	    swig::SwigVar_PyObject key = swig::from(i->first);
	    swig::SwigVar_PyObject val = swig::from(i->second);
	    PyDict_SetItem(obj, key, val);
	  }
	  return obj;
	}
	*/
	return octave_value();
      }
    };
  }
}

%define %swig_map_common(Map...)
  %swig_sequence_iterator(Map);
  %swig_container_methods(Map);
%enddef

%define %swig_map_methods(Map...)
     %swig_map_common(Map)
%enddef


%include <std/std_map.i>
