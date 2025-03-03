%fragment("StdTraits","header",fragment="StdTraitsCommon")
{
namespace swig {
// Traits that provides the from method
  template <class Type> struct traits_from_ptr {
    static SwigSciObject from(Type *val, int owner = 0) {
        return SWIG_OK; //SWIG_NewPointerObj(val, type_info<Type>(), owner);
    }
  };

  template <class Type> struct traits_from {
    static SwigSciObject from(const Type& val) {
      return traits_from_ptr<Type>::from(new Type(val), 1);
    }
  };

  template <class Type> struct traits_from<Type *> {
    static SwigSciObject from(Type* val) {
      return traits_from_ptr<Type>::from(val, 0);
    }
  };

  template <class Type> struct traits_from<const Type *> {
    static SwigSciObject from(const Type* val) {
      return traits_from_ptr<Type>::from(const_cast<Type*>(val), 0);
    }
  };


  template <class Type>
  inline SwigSciObject from(const Type& val) {
    return traits_from<Type>::from(val);
  }

  template <class Type>
  inline SwigSciObject from_ptr(Type* val, int owner) {
    return traits_from_ptr<Type>::from(val, owner);
  }

    // Traits that provides the asval/as/check method
  template <class Type>
  struct traits_asptr {
    static int asptr(const SwigSciObject& obj, Type **val) {
      Type *p = 0;
      swig_type_info *descriptor = type_info<Type>();
      int res = descriptor ? SWIG_ConvertPtr(obj, (void **)&p, descriptor, 0) : SWIG_ERROR;
      if (SWIG_IsOK(res)) {
	if (val) *val = p;
      }
      return res;
    }
  };

  template <class Type>
  inline int asptr(const SwigSciObject& obj, Type **vptr) {
    return traits_asptr<Type>::asptr(obj, vptr);
  }

  template <class Type>
  struct traits_asval {
    static int asval(const SwigSciObject& obj, Type *val) {
      if (val) {
	Type *p = 0;
	int res = traits_asptr<Type>::asptr(obj, &p);
	if (!SWIG_IsOK(res))
    return res;
	if (p) {
	  typedef typename noconst_traits<Type>::noconst_type noconst_type;
	  *(const_cast<noconst_type*>(val)) = *p;
	  if (SWIG_IsNewObj(res)){
	    %delete(p);
	    res = SWIG_DelNewMask(res);
	  }
	  return res;
	} else {
	  return SWIG_ERROR;
	}
      } else {
	return traits_asptr<Type>::asptr(obj, (Type **)(0));
      }
    }
  };

  template <class Type> struct traits_asval<Type*> {
    static int asval(const SwigSciObject& obj, Type **val) {
      if (val) {
        typedef typename noconst_traits<Type>::noconst_type noconst_type;
        noconst_type *p = 0;
        int res = traits_asptr<noconst_type>::asptr(obj,  &p);
        if (SWIG_IsOK(res)) {
          *(const_cast<noconst_type**>(val)) = p;
	}
	return res;
      } else {
	return traits_asptr<Type>::asptr(obj, (Type **)(0));
      }
    }
  };

  template <class Type>
  inline int asval(const SwigSciObject& obj, Type *val) {
    return traits_asval<Type>::asval(obj, val);
  }

  template <class Type>
  struct traits_as<Type, value_category> {
    static Type as(const SwigSciObject& obj) {
      Type v;
      int res = asval(obj, &v);
      if (SWIG_IsOK(res)) {
        return v;
      } else {
        %type_error(swig::type_name<Type>());
        throw std::invalid_argument("bad type");
      }
    }
  };

  template <class Type>
  struct traits_as<Type, pointer_category> {
    static Type as(const SwigSciObject& obj) {
      Type *v = 0;
      int res = traits_asptr<Type>::asptr(obj, &v);
      if (SWIG_IsOK(res) && v) {
	if (SWIG_IsNewObj(res)) {
	  Type r(*v);
	  %delete(v);
	  return r;
	} else {
	  return *v;
	}
      } else {
        %type_error(swig::type_name<Type>());
        throw std::invalid_argument("bad type");
      }
    }
  };

  template <class Type>
  struct traits_as<Type*, pointer_category> {
    static Type* as(const SwigSciObject& obj) {
      Type *v = 0;
      int res = traits_asptr<Type>::asptr(obj, &v);
      if (SWIG_IsOK(res)) {
	return v;
      } else {
        %type_error(swig::type_name<Type>());
        throw std::invalid_argument("bad type");
      }
    }
  };

  template <class Type>
  inline Type as(const SwigSciObject& obj) {
    return traits_as<Type, typename traits<Type>::category>::as(obj);
  }

  template <class Type>
  struct traits_check<Type, value_category> {
    static bool check(const SwigSciObject& obj) {
      int res = asval(obj, (Type *)(0));
      return SWIG_IsOK(res) ? true : false;
    }
  };

  template <class Type>
  struct traits_check<Type, pointer_category> {
    static bool check(const SwigSciObject& obj) {
      int res = asptr(obj, (Type **)(0));
      return SWIG_IsOK(res) ? true : false;
    }
  };

  template <class Type>
  inline bool check(const SwigSciObject& obj) {
    return traits_check<Type, typename traits<Type>::category>::check(obj);
  }
}
}

%define %specialize_std_container(Type,Check,As,From)
%{
namespace swig {
  template <>  struct traits_asval<Type > {
    typedef Type value_type;
    static int asval(const SwigSciObject& obj, value_type *val) {
      if (Check(obj)) {
	if (val) *val = As(obj);
	return SWIG_OK;
      }
      return SWIG_ERROR;
    }
  };
  template <>  struct traits_from<Type > {
    typedef Type value_type;
    static SwigSciObject from(const value_type& val) {
      return From(val);
    }
  };

  template <>
  struct traits_check<Type, value_category> {
    static int check(const SwigSciObject& obj) {
      int res = Check(obj);
      return obj && res ? res : 0;
    }
  };
}
%}
%enddef


#define specialize_std_vector(Type,Check,As,From) %specialize_std_container(%arg(Type),Check,As,From)
#define specialize_std_list(Type,Check,As,From) %specialize_std_container(%arg(Type),Check,As,From)
#define specialize_std_deque(Type,Check,As,From) %specialize_std_container(%arg(Type),Check,As,From)
#define specialize_std_set(Type,Check,As,From) %specialize_std_container(%arg(Type),Check,As,From)
#define specialize_std_multiset(Type,Check,As,From) %specialize_std_container(%arg(Type),Check,As,From)

