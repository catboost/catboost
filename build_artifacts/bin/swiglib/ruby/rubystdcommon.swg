
/* ------------------------------------------------------------
 * The Ruby classes, for C++
 * ------------------------------------------------------------ */
%include <rubyclasses.swg>
%include <rubystdcommon_forward.swg>

%fragment("StdTraits","header",fragment="StdTraitsCommon",fragment="StdTraitsForwardDeclaration")
{

namespace swig {  
  /*
    Traits that provides the from method
  */
  template <class Type> struct traits_from_ptr {
    static VALUE from(Type *val, int owner = 0) {
      return SWIG_NewPointerObj(val, type_info<Type>(), owner);
    }
  };

  template <class Type> struct traits_from {
    static VALUE from(const Type& val) {
      return traits_from_ptr<Type>::from(new Type(val), 1);
    }
  };

  template <class Type> struct traits_from<Type *> {
    static VALUE from(Type* val) {
      return traits_from_ptr<Type>::from(val, 0);
    }
  };

  template <class Type> struct traits_from<const Type *> {
    static VALUE from(const Type* val) {
      return traits_from_ptr<Type>::from(const_cast<Type*>(val), 0);
    }
  };


  template <class Type>
  inline VALUE from(const Type& val) {
    return traits_from<Type>::from(val);
  }

  template <class Type>
  inline VALUE from_ptr(Type* val, int owner) {
    return traits_from_ptr<Type>::from(val, owner);
  }

  /*
    Traits that provides the asval/as/check method
  */
  template <class Type>
  struct traits_asptr {   
    static int asptr(VALUE obj, Type **val) {
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
  inline int asptr(VALUE obj, Type **vptr) {
    return traits_asptr<Type>::asptr(obj, vptr);
  }

  template <class Type> 
  struct traits_asval {
    static int asval(VALUE obj, Type *val) {
      if (val) {
	Type *p = 0;
	int res = traits_asptr<Type>::asptr(obj, &p);
	if (!SWIG_IsOK(res)) return res;	
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
    static int asval(VALUE obj, Type **val) {
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
  inline int asval(VALUE obj, Type *val) {
    return traits_asval<Type>::asval(obj, val);
  }

  template <class Type> 
  struct traits_as<Type, value_category> {
    static Type as(VALUE obj) {
      Type v;
      int res = asval(obj, &v);
      if (!SWIG_IsOK(res)) {
	VALUE lastErr = rb_gv_get("$!");
	if (lastErr == Qnil) {
	  %type_error(swig::type_name<Type>());
	}
        throw std::invalid_argument("bad type");
      }
      return v;
    }
  };

  template <class Type> 
  struct traits_as<Type, pointer_category> {
    static Type as(VALUE obj) {
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
	VALUE lastErr = rb_gv_get("$!");
	if (lastErr == Qnil) {
	  %type_error(swig::type_name<Type>());
	}
        throw std::invalid_argument("bad type");
      }
    }
  };

  template <class Type> 
  struct traits_as<Type*, pointer_category> {
    static Type* as(VALUE obj) {
      Type *v = 0;      
      int res = traits_asptr<Type>::asptr(obj, &v);
      if (SWIG_IsOK(res)) {
	return v;
      } else {
	VALUE lastErr = rb_gv_get("$!");
	if (lastErr == Qnil) {
	  %type_error(swig::type_name<Type>());
	}
        throw std::invalid_argument("bad type");
      }
    }
  };

  template <class Type>
  inline Type as(VALUE obj) {
    return traits_as< Type, typename traits< Type >::category >::as(obj);
  }

  template <class Type> 
  struct traits_check<Type, value_category> {
    static bool check(VALUE obj) {
      int res = asval(obj, (Type *)(0));
      return SWIG_IsOK(res) ? true : false;
    }
  };

  template <class Type> 
  struct traits_check<Type, pointer_category> {
    static bool check(VALUE obj) {
      int res = asptr(obj, (Type **)(0));
      return SWIG_IsOK(res) ? true : false;
    }
  };

  template <class Type>
  inline bool check(VALUE obj) {
    return traits_check<Type, typename traits<Type>::category>::check(obj);
  }
}
}

