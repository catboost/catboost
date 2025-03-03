#define SWIG_SHARED_PTR_NAMESPACE std
%include <boost_shared_ptr.i>
%include <rubystdcommon_forward.swg>


%fragment("StdSharedPtrTraits","header",fragment="StdTraitsForwardDeclaration",fragment="<memory>")
{
namespace swig {
  /*
   Template specialization for functions defined in rubystdcommon.swg. Special handling for shared_ptr
   is required as, shared_ptr<T> * is used rather than the usual T *, see shared_ptr.i.
  */
  template <class Type>
  struct traits_asptr<std::shared_ptr<Type> > {
    static int asptr(VALUE obj, std::shared_ptr<Type> **val) {
      int res = SWIG_ERROR;
      swig_type_info *descriptor = type_info<std::shared_ptr<Type> >();
      if (val) {
        std::shared_ptr<Type> *p = 0;
        swig_ruby_owntype newmem = {0, 0};
        res = descriptor ? SWIG_ConvertPtrAndOwn(obj, (void **)&p, descriptor, 0, &newmem) : SWIG_ERROR;
        if (SWIG_IsOK(res)) {
          if (*val) {
            **val = p ? *p : std::shared_ptr<Type>();
          } else {
            *val = p;
            if (newmem.own & SWIG_CAST_NEW_MEMORY) {
              // Upcast for pointers to shared_ptr in this generic framework has not been implemented
              res = SWIG_ERROR;
            }
          }
          if (newmem.own & SWIG_CAST_NEW_MEMORY)
            delete p;
        }
      } else {
        res = descriptor ? SWIG_ConvertPtr(obj, 0, descriptor, 0) : SWIG_ERROR;
      }
      return res;
    }
  };

  template <class Type>
  struct traits_asval<std::shared_ptr<Type> > {
    static int asval(VALUE obj, std::shared_ptr<Type> *val) {
      if (val) {
	std::shared_ptr<Type> ret;
	std::shared_ptr<Type> *p = &ret;
	int res = traits_asptr<std::shared_ptr<Type> >::asptr(obj, &p);
	if (!SWIG_IsOK(res))
	  return res;
	*val = ret;
	return SWIG_OK;
      } else {
	return traits_asptr<std::shared_ptr<Type> >::asptr(obj, (std::shared_ptr<Type> **)(0));
      }
    }
  };

  template <class Type>
    struct traits_asval<std::shared_ptr<Type> *> {
    static int asval(VALUE obj, std::shared_ptr<Type> **val) {
      if (val) {
	typedef typename noconst_traits<std::shared_ptr<Type> >::noconst_type noconst_type;
	if (*val) {
	  noconst_type ret;
	  noconst_type *p = &ret;
	  int res = traits_asptr<noconst_type>::asptr(obj, &p);
	  if (SWIG_IsOK(res))
	    **(const_cast<noconst_type**>(val)) = ret;
	  return res;
	} else {
	  noconst_type *p = 0;
	  int res = traits_asptr<noconst_type>::asptr(obj,  &p);
	  if (SWIG_IsOK(res))
	    *val = p;
	  return res;
	}
      } else {
	return traits_asptr<std::shared_ptr<Type> >::asptr(obj, (std::shared_ptr<Type> **)(0));
      }
    }
  };

  template <class Type>
  struct traits_as<std::shared_ptr<Type>, pointer_category> {
    static std::shared_ptr<Type> as(VALUE obj) {
      std::shared_ptr<Type> ret;
      std::shared_ptr<Type> *v = &ret;
      int res = traits_asptr<std::shared_ptr<Type> >::asptr(obj, &v);
      if (SWIG_IsOK(res)) {
	return ret;
      } else {
	
	VALUE lastErr = rb_gv_get("$!");
	if (lastErr == Qnil)
	  SWIG_Error(SWIG_TypeError,  swig::type_name<std::shared_ptr<Type> >());
        throw std::invalid_argument("bad type");
      }
    }
  };

  template <class Type>
  struct traits_as<std::shared_ptr<Type> *, pointer_category> {
    static std::shared_ptr<Type> * as(VALUE obj) {
      std::shared_ptr<Type> *p = 0;
      int res = traits_asptr<std::shared_ptr<Type> >::asptr(obj, &p);
      if (SWIG_IsOK(res)) {
	return p;
      } else {
	
	VALUE lastErr = rb_gv_get("$!");
	if (lastErr == Qnil)
	  SWIG_Error(SWIG_TypeError,  swig::type_name<std::shared_ptr<Type> *>());
        throw std::invalid_argument("bad type");
      }
    }
  };

  template <class Type>
  struct traits_from_ptr<std::shared_ptr<Type> > {
    static VALUE from(std::shared_ptr<Type> *val, int owner = 0) {
      if (val && *val) {
        return SWIG_NewPointerObj(val, type_info<std::shared_ptr<Type> >(), owner);
      } else {
        return Qnil;
      }
    }
  };

  /*
   The descriptors in the shared_ptr typemaps remove the const qualifier for the SWIG type system.
   Remove const likewise here, otherwise SWIG_TypeQuery("std::shared_ptr<const Type>") will return NULL.
  */
  template<class Type>
  struct traits_from<std::shared_ptr<const Type> > {
    static VALUE from(const std::shared_ptr<const Type>& val) {
      std::shared_ptr<Type> p = std::const_pointer_cast<Type>(val);
      return swig::from(p);
    }
  };
}
}

%fragment("StdSharedPtrTraits");
