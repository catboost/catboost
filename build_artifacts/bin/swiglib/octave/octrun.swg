#if !SWIG_OCTAVE_PREREQ(3,2,0)
#define SWIG_DEFUN(cname, wname, doc) DEFUNX_DLD(#cname, wname, FS ## cname, args, nargout, doc)
#else
#define SWIG_DEFUN(cname, wname, doc) DEFUNX_DLD(#cname, wname, G ## cname, args, nargout, doc)
#endif

SWIGRUNTIME bool SWIG_check_num_args(const char *func_name, int num_args, int max_args, int min_args, int varargs) {
  if (num_args > max_args && !varargs)
    error("function %s takes at most %i arguments", func_name, max_args);
  else if (num_args < min_args)
    error("function %s requires at least %i arguments", func_name, min_args);
  else
    return true;
  return false;
}

SWIGRUNTIME octave_value_list *SWIG_Octave_AppendOutput(octave_value_list *ovl, const octave_value &ov) {
  ovl->append(ov);
  return ovl;
}

SWIGRUNTIME octave_value SWIG_ErrorType(int code) {
  switch (code) {
  case SWIG_MemoryError:
    return "SWIG_MemoryError";
  case SWIG_IOError:
    return "SWIG_IOError";
  case SWIG_RuntimeError:
    return "SWIG_RuntimeError";
  case SWIG_IndexError:
    return "SWIG_IndexError";
  case SWIG_TypeError:
    return "SWIG_TypeError";
  case SWIG_DivisionByZero:
    return "SWIG_DivisionByZero";
  case SWIG_OverflowError:
    return "SWIG_OverflowError";
  case SWIG_SyntaxError:
    return "SWIG_SyntaxError";
  case SWIG_ValueError:
    return "SWIG_ValueError";
  case SWIG_SystemError:
    return "SWIG_SystemError";
  case SWIG_AttributeError:
    return "SWIG_AttributeError";
  }
  return "SWIG unknown error";
}

SWIGRUNTIME octave_value SWIG_Error(int code, const char *msg) {
  octave_value type(SWIG_ErrorType(code));
  std::string r = msg;
  r += " (" + type.string_value() + ")";
  error("%s", r.c_str());
  return octave_value(r);
}

#define SWIG_fail                                       goto fail

#define SWIG_Octave_ConvertPtr(obj, pptr, type, flags)  SWIG_Octave_ConvertPtrAndOwn(obj, pptr, type, flags, 0)
#define SWIG_ConvertPtr(obj, pptr, type, flags)         SWIG_Octave_ConvertPtr(obj, pptr, type, flags)
#define SWIG_ConvertPtrAndOwn(obj,pptr,type,flags,own)  SWIG_Octave_ConvertPtrAndOwn(obj, pptr, type, flags, own)
#define SWIG_ConvertPtr(obj, pptr, type, flags)         SWIG_Octave_ConvertPtr(obj, pptr, type, flags)
#define SWIG_NewPointerObj(ptr, type, flags)            SWIG_Octave_NewPointerObj(ptr, type, flags)
#define swig_owntype                                    int

#define SWIG_ConvertPacked(obj, ptr, sz, ty)            SWIG_Octave_ConvertPacked(obj, ptr, sz, ty)
#define SWIG_NewPackedObj(ptr, sz, type)                SWIG_Octave_NewPackedObj(ptr, sz, type)

#define SWIG_ConvertFunctionPtr(obj, pptr, type)        SWIG_ConvertPtr(obj, pptr, type, 0)
#define SWIG_NewFunctionPtrObj(ptr, type)               SWIG_NewPointerObj(ptr, type, 0)

#define SWIG_ConvertMember(obj, ptr, sz, ty)            SWIG_Octave_ConvertPacked(obj, ptr, sz, ty)
#define SWIG_NewMemberObj(ptr, sz, type)                SWIG_Octave_NewPackedObj(ptr, sz, type)

#define SWIG_GetModule(clientdata) SWIG_Octave_GetModule(clientdata)
#define SWIG_SetModule(clientdata, pointer) SWIG_Octave_SetModule(clientdata,pointer);
#define SWIG_MODULE_CLIENTDATA_TYPE void*

#define Octave_Error_Occurred() 0
#define SWIG_Octave_AddErrorMsg(msg) {;}

SWIGRUNTIME swig_module_info *SWIG_Octave_GetModule(void *clientdata);
SWIGRUNTIME void SWIG_Octave_SetModule(void *clientdata, swig_module_info *pointer);

// For backward compatibility only
#define SWIG_POINTER_EXCEPTION  0
#define SWIG_arg_fail(arg)      0

// Runtime API implementation

typedef octave_value_list(*octave_func) (const octave_value_list &, int);
class octave_swig_type;

namespace Swig {

#ifdef SWIG_DIRECTORS

  class Director;

  typedef std::map < void *, Director * > rtdir_map;
  SWIGINTERN rtdir_map* get_rtdir_map();
  SWIGINTERNINLINE void set_rtdir(void *vptr, Director *d);
  SWIGINTERNINLINE void erase_rtdir(void *vptr);
  SWIGINTERNINLINE Director *get_rtdir(void *vptr);

  SWIGRUNTIME void swig_director_destroyed(octave_swig_type *self, Director *d);
  SWIGRUNTIME octave_swig_type *swig_director_get_self(Director *d);
  SWIGRUNTIME void swig_director_set_self(Director *d, octave_swig_type *self);

#endif

  SWIGRUNTIME octave_base_value *swig_value_ref(octave_swig_type *ost);
  SWIGRUNTIME octave_swig_type *swig_value_deref(octave_value ov);
  SWIGRUNTIME octave_swig_type *swig_value_deref(const octave_base_value &ov);
}

#ifdef SWIG_DIRECTORS
SWIGRUNTIME void swig_acquire_ownership(void *vptr);
SWIGRUNTIME void swig_acquire_ownership_array(void *vptr);
SWIGRUNTIME void swig_acquire_ownership_obj(void *vptr, int own);
#endif

  struct swig_octave_member {
    const char *name;
    octave_func method;
    octave_func get_method;
    octave_func set_method;
    int flags;			// 1 static, 2 global
    const char *doc;
    bool is_static() const {
      return flags &1;
    } bool is_global() const {
      return flags &2;
    }
  };

  struct swig_octave_class {
    const char *name;
    swig_type_info **type;
    int director;
    octave_func constructor;
    const char *constructor_doc;
    octave_func destructor;
    const swig_octave_member *members;
    const char **base_names;
    const swig_type_info **base;
  };

#if SWIG_OCTAVE_PREREQ(4,4,0)
  // in Octave 4.4 behaviour of octave_builtin() appears to have changed and 'self' argument is no longer passed
  // to function (maybe because this is now a 'method'??) so need to create our own octave_function subclass
#define SWIG_OCTAVE_BOUND_FUNC(func, args) octave_value(new octave_swig_bound_func(func, args))
  class octave_swig_bound_func : public octave_function {
  public:

    octave_swig_bound_func(void) : octave_function(), method(0), first_args()
      { }

    octave_swig_bound_func(octave_function* _method, octave_value_list _first_args)
      : octave_function("", ""), method(_method), first_args(_first_args)
      { }

    octave_swig_bound_func(const octave_swig_bound_func& f) = delete;

    octave_swig_bound_func& operator= (const octave_swig_bound_func& f) = delete;

    ~octave_swig_bound_func(void) = default;

    bool is_function(void) const { return true; }

    octave_function* function_value(bool = false) { return this; }

    octave_value_list call(octave::tree_evaluator& tw, int nargout = 0, const octave_value_list& args = octave_value_list()) {
      octave_value_list all_args;
      all_args.append(first_args);
      all_args.append(args);
      return method->call(tw, nargout, all_args);
    }

    octave_value subsref(const std::string &ops, const std::list < octave_value_list > &idx) {
      octave_value_list ovl = subsref(ops, idx, 1);
      return ovl.length() ? ovl(0) : octave_value();
    }

    octave_value_list subsref(const std::string &ops, const std::list < octave_value_list > &idx, int nargout) {
      assert(ops.size() > 0);
      assert(ops.size() == idx.size());
      if (ops != "(")
        error("invalid function call");
      octave::tree_evaluator& tw = octave::interpreter::the_interpreter()->get_evaluator();
      return call(tw, nargout, *idx.begin());
    }

  protected:

    octave_function* method;
    octave_value_list first_args;

    std::set<std::string> dispatch_classes;

  };
#else
#define SWIG_OCTAVE_BOUND_FUNC(func, args) octave_value(func)
#endif

  // octave_swig_type plays the role of both the shadow class and the class 
  // representation within Octave, since there is no support for classes.
  //
  // These should really be decoupled, with the class support added to Octave
  // and the shadow class given by an m-file script. That would dramatically 
  // reduce the runtime complexity, and be more in line w/ other modules.

  class octave_swig_type:public octave_base_value {
    struct cpp_ptr {
      void *ptr;
      bool destroyed;
      cpp_ptr(void *_ptr):ptr(_ptr), destroyed(false) {
      }};
    typedef std::pair < const swig_type_info *, cpp_ptr > type_ptr_pair;

    mutable swig_module_info *module;

    const swig_type_info *construct_type;	// type of special type object
    std::vector < type_ptr_pair > types;	// our c++ base classes
    int own;			// whether we call c++ destructors when we die

    typedef std::pair < const swig_octave_member *, octave_value > member_value_pair;
    typedef std::map < std::string, member_value_pair > member_map;
    member_map members;
    bool always_static;

    const swig_octave_member *find_member(const swig_type_info *type, const std::string &name) {
      if (!type->clientdata)
	return 0;
      swig_octave_class *c = (swig_octave_class *) type->clientdata;
      const swig_octave_member *m;
      for (m = c->members; m->name; ++m)
	if (m->name == name)
	  return m;
      for (int j = 0; c->base_names[j]; ++j) {
	if (!c->base[j]) {
	  if (!module)
	    module = SWIG_GetModule(0);
	  assert(module);
	  c->base[j] = SWIG_MangledTypeQueryModule(module, module, c->base_names[j]);
	}
	if (!c->base[j])
	  return 0;
	if ((m = find_member(c->base[j], name)))
	  return m;
      }
      return 0;
    }

    member_value_pair *find_member(const std::string &name, bool insert_if_not_found) {
      member_map::iterator it = members.find(name);
      if (it != members.end())
	return &it->second;
      const swig_octave_member *m;
      for (unsigned int j = 0; j < types.size(); ++j)
	if ((m = find_member(types[j].first, name)))
	  return &members.insert(std::make_pair(name, std::make_pair(m, octave_value()))).first->second;
      if (!insert_if_not_found)
	return 0;
      return &members[name];
    }

    const swig_type_info *find_base(const std::string &name, const swig_type_info *base) {
      if (!base) {
	for (unsigned int j = 0; j < types.size(); ++j) {
	  assert(types[j].first->clientdata);
	  swig_octave_class *cj = (swig_octave_class *) types[j].first->clientdata;
	  if (cj->name == name)
	    return types[j].first;
	}
	return 0;
      }
      assert(base->clientdata);
      swig_octave_class *c = (swig_octave_class *) base->clientdata;
      for (int j = 0; c->base_names[j]; ++j) {
	if (!c->base[j]) {
	  if (!module)
	    module = SWIG_GetModule(0);
	  assert(module);
	  c->base[j] = SWIG_MangledTypeQueryModule(module, module, c->base_names[j]);
	}
	if (!c->base[j])
	  return 0;
	assert(c->base[j]->clientdata);
	swig_octave_class *cj = (swig_octave_class *) c->base[j]->clientdata;
	if (cj->name == name)
	  return c->base[j];
      }
      return 0;
    }

    void load_members(const swig_octave_class* c,member_map& out) const {
      for (const swig_octave_member *m = c->members; m->name; ++m) {
	if (out.find(m->name) == out.end())
	  out.insert(std::make_pair(m->name, std::make_pair(m, octave_value())));
      }
      for (int j = 0; c->base_names[j]; ++j) {
	if (!c->base[j]) {
	  if (!module)
	    module = SWIG_GetModule(0);
	  assert(module);
	  c->base[j] = SWIG_MangledTypeQueryModule(module, module, c->base_names[j]);
	}
	if (!c->base[j])
	  continue;
	assert(c->base[j]->clientdata);
	const swig_octave_class *cj =
	  (const swig_octave_class *) c->base[j]->clientdata;
	load_members(cj,out);
      }
    }

    void load_members(member_map& out) const {
      out=members;
      for (unsigned int j = 0; j < types.size(); ++j)
	if (types[j].first->clientdata)
	  load_members((const swig_octave_class *) types[j].first->clientdata, out);
    }

    octave_value_list member_invoke(member_value_pair *m, const octave_value_list &args, int nargout) {
      if (m->second.is_defined())
	return m->second.subsref("(", std::list < octave_value_list > (1, args), nargout);
      else if (m->first && m->first->method)
	return m->first->method(args, nargout);
      error("member not defined or not invocable");
      return octave_value_list();
    }

    bool dispatch_unary_op(const std::string &symbol, octave_value &ret) const {
      octave_swig_type *nc_this = const_cast < octave_swig_type *>(this);
      member_value_pair *m = nc_this->find_member(symbol, false);
      if (!m || m->first->is_static() || m->first->is_global())
	return false;
      octave_value_list args;
      args.append(nc_this->as_value());
      octave_value_list argout(nc_this->member_invoke(m, args, 1));
      if (argout.length() < 1)
	return false;
      ret = argout(0);
      return true;
    }

    bool dispatch_binary_op(const std::string &symbol, const octave_base_value &rhs, octave_value &ret) const {
      octave_swig_type *nc_this = const_cast < octave_swig_type *>(this);
      member_value_pair *m = nc_this->find_member(symbol, false);
      if (!m || m->first->is_static() || m->first->is_global())
	return false;
      octave_value_list args;
      args.append(nc_this->as_value());
      args.append(make_value_hack(rhs));
      octave_value_list argout(nc_this->member_invoke(m, args, 1));
      if (argout.length() < 1)
	return false;
      ret = argout(0);
      return true;
    }

    bool dispatch_index_op(const std::string &symbol, const octave_value_list &rhs, octave_value_list &ret) const {
      octave_swig_type *nc_this = const_cast < octave_swig_type *>(this);
      member_value_pair *m = nc_this->find_member(symbol, false);
      if (!m || m->first->is_static() || m->first->is_global())
	return false;
      octave_value_list args;
      args.append(nc_this->as_value());
      args.append(rhs);
      octave_value_list argout(nc_this->member_invoke(m, args, 1));
      if (argout.length() >= 1)
	ret = argout(0);
      return true;
    }

    octave_value_list member_deref(member_value_pair *m, const octave_value_list &args) {
      if (m->second.is_defined()) {
        if (m->second.is_function() || m->second.is_function_handle()) {
          return SWIG_OCTAVE_BOUND_FUNC(m->second.function_value(), args);
        } else {
          return m->second;
        }
      } else if (m->first) {
	if (m->first->get_method)
	  return m->first->get_method(args, 1);
	else if (m->first->method)
          return SWIG_OCTAVE_BOUND_FUNC(new octave_builtin(m->first->method), args);
      }
      error("undefined member");
      return octave_value_list();
    }

    static octave_value make_value_hack(const octave_base_value &x) {
      ((octave_swig_type &) x).count++;
      return octave_value((octave_base_value *) &x);
    }

    octave_swig_type(const octave_swig_type &x);
    octave_swig_type &operator=(const octave_swig_type &rhs);
  public:

    octave_swig_type(void *_ptr = 0, const swig_type_info *_type = 0, int _own = 0,
		     bool _always_static = false)
      :	module(0), construct_type(_ptr ? 0 : _type), own(_own), 
      always_static(_always_static) {
      if (_type || _ptr)
	types.push_back(std::make_pair(_type, _ptr));
#ifdef SWIG_DIRECTORS
      if (_ptr) {
	Swig::Director *d = Swig::get_rtdir(_ptr);
	if (d)
	  Swig::swig_director_set_self(d, this);
      }
#endif
    }

    ~octave_swig_type() {
      if (own) {
	++count;
	for (unsigned int j = 0; j < types.size(); ++j) {
	  if (!types[j].first || !types[j].first->clientdata)
	    continue;
	  swig_octave_class *c = (swig_octave_class *) types[j].first->clientdata;
	  if (c->destructor && !types[j].second.destroyed && types[j].second.ptr) {
	    c->destructor(as_value(), 0);
	  }
	}
      }
#ifdef SWIG_DIRECTORS
      for (unsigned int j = 0; j < types.size(); ++j)
	Swig::erase_rtdir(types[j].second.ptr);
#endif
    }

    dim_vector dims(void) const {
      octave_value out;
      if (!dispatch_unary_op("__dims__", out))
        return dim_vector(1,1);

      // Return value should be cell or matrix of integers
#if SWIG_OCTAVE_PREREQ(4,4,0)
      if (out.iscell()) {
#else
      if (out.is_cell()) {
#endif
        const Cell & c=out.cell_value();
        int ndim = c.rows();
        if (ndim==1 && c.columns()!=1) ndim = c.columns();

        dim_vector d;
        d.resize(ndim < 2 ? 2 : ndim);
        d(0) = d(1) = 1;

        // Fill in dim_vector 
        for (int k=0;k<ndim;k++) {
          const octave_value& obj = c(k);
          d.elem(k) = obj.int_value();
          
          // __dims__ should return a cell filled with integers
          if (error_state) return dim_vector(1,1);
        }
        return d;
#if SWIG_OCTAVE_PREREQ(4,4,0)
      } else if (out.is_matrix_type() || out.isnumeric() ) {
#else
      } else if (out.is_matrix_type() || out.is_numeric_type() ) {
#endif
        if (out.rows()==1 || out.columns()==1) {
           Array<int> a = out.int_vector_value();
           if (error_state) return dim_vector(1,1);
           dim_vector d;
           d.resize(a.numel() < 2 ? 2 : a.numel());
           d(0) = d(1) = 1;
           for (int k=0;k<a.numel();k++) {
              d.elem(k) = a(k);
           }
           return d;
        } else {
          return dim_vector(1,1);
        }
      } else {
        return dim_vector(1,1);
      }
    }

    octave_value as_value() {
      ++count;
      return Swig::swig_value_ref(this);
    }

    void incref() {
      ++count;
    }

    void decref() {
      if (!--count)
	delete this;
    }

    size_t swig_this() const {
      if (!types.size())
	return (size_t) this;
      return (size_t) types[0].second.ptr;
    }
    const char* help_text() const {
      if (!types.size())
	return 0;
      if (!types[0].first->clientdata)
	return 0;
      swig_octave_class *c = (swig_octave_class *) types[0].first->clientdata;
      return c->constructor_doc;
    }

    std::string swig_type_name() const {
      // * need some way to manually name subclasses.
      // * eg optional first arg to subclass(), or named_subclass()
      std::string ret;
      for (unsigned int j = 0; j < types.size(); ++j) {
	if (j)
	  ret += "_";
	if (types[j].first->clientdata) {
	  swig_octave_class *c = (swig_octave_class *) types[j].first->clientdata;
	  ret += c->name;
	} else
	  ret += types[j].first->name;
      }
      return ret;
    }

    void merge(octave_swig_type &rhs) {
      rhs.own = 0;
      for (unsigned int j = 0; j < rhs.types.size(); ++j) {
	assert(!rhs.types[j].second.destroyed);
#ifdef SWIG_DIRECTORS
	Swig::Director *d = Swig::get_rtdir(rhs.types[j].second.ptr);
	if (d)
	  Swig::swig_director_set_self(d, this);
#endif
      }
      types.insert(types.end(), rhs.types.begin(), rhs.types.end());
      members.insert(rhs.members.begin(), rhs.members.end());
#if SWIG_OCTAVE_PREREQ(4,4,0)
      assign(rhs.swig_type_name(), rhs.as_value());
#else
      rhs.types.clear();
      rhs.members.clear();
#endif
    }

    typedef member_map::const_iterator swig_member_const_iterator;
    swig_member_const_iterator swig_members_begin() { return members.begin(); }
    swig_member_const_iterator swig_members_end() { return members.end(); }

    int cast(void **vptr, swig_type_info *type, int *_own, int flags) {
      int res = SWIG_ERROR;
      if (_own)
	*_own = own;
      if (flags &SWIG_POINTER_DISOWN)
	own = 0;
      if (!type && types.size()) {
	if(vptr)
          *vptr = types[0].second.ptr;
        return SWIG_OK;
      }
      for (unsigned int j = 0; j < types.size(); ++j)
	if (type == types[j].first) {
	  if(vptr)
            *vptr = types[j].second.ptr;
          return SWIG_OK;
        }
      for (unsigned int j = 0; j < types.size(); ++j) {
	swig_cast_info *tc = SWIG_TypeCheck(types[j].first->name, type);
	if (!tc)
	  continue;
        if(vptr) {
	  int newmemory = 0;
	  *vptr = SWIG_TypeCast(tc, types[j].second.ptr, &newmemory);
	    if (newmemory == SWIG_CAST_NEW_MEMORY) {
              assert(_own); /* badly formed typemap which will lead to a memory leak - it must set and use own to delete *ptr */
              if (_own)
                *_own = *_own | SWIG_CAST_NEW_MEMORY;
            }
        }
        res = SWIG_OK;
        break;
      }
      return res;
    }

    bool is_owned() const {
      return own;
    }

#ifdef SWIG_DIRECTORS
    void director_destroyed(Swig::Director *d) {
      bool found = false;
      for (unsigned int j = 0; j < types.size(); ++j) {
	Swig::Director *dj = Swig::get_rtdir(types[j].second.ptr);
	if (dj == d) {
	  types[j].second.destroyed = true;
	  found = true;
	}
      }
      assert(found);
    }
#endif

    void assign(const std::string &name, const octave_value &ov) {
      members[name] = std::make_pair((const swig_octave_member *) 0, ov);
    }

    void assign(const std::string &name, const swig_octave_member *m) {
      members[name] = std::make_pair(m, octave_value());
    }

    octave_base_value *clone() const {
      // pass-by-value is probably not desired, and is harder;
      // requires calling copy constructors of contained types etc.
      assert(0);
      *(int *) 0 = 0;
      return 0;
    }

    octave_base_value *empty_clone() const {
      return new octave_swig_type();
    }

    bool is_defined() const {
      return true;
    }

    virtual bool is_map() const {
      return true;
    }

    virtual octave_value subsref(const std::string &ops, const std::list < octave_value_list > &idx) {
      octave_value_list ovl = subsref(ops, idx, 1);
      return ovl.length()? ovl(0) : octave_value();
    }

    virtual octave_value_list subsref(const std::string &ops, const std::list < octave_value_list > &idx, int nargout) {
      assert(ops.size() > 0);
      assert(ops.size() == idx.size());

      std::list < octave_value_list >::const_iterator idx_it = idx.begin();
      int skip = 0;
      octave_value_list sub_ovl;

      // constructor invocation
      if (ops[skip] == '(' && construct_type) {
	assert(construct_type->clientdata);
	swig_octave_class *c = (swig_octave_class *) construct_type->clientdata;
	if (!c->constructor) {
	  error("cannot create instance");
	  return octave_value_list();
	}
	octave_value_list args;
	if (c->director)
	  args.append(Swig::swig_value_ref(new octave_swig_type(this, 0, 0)));
	args.append(*idx_it++);
	++skip;
	sub_ovl = c->constructor(args, nargout);
      }
      // member dereference or invocation
      else if (ops[skip] == '.') {
	std::string subname;
	const swig_type_info *base = 0;	// eg, a.base.base_cpp_mem
	for (;;) {
	  octave_value_list subname_ovl(*idx_it++);
	  ++skip;
	  assert(subname_ovl.length() == 1 && subname_ovl(0).is_string());
	  subname = subname_ovl(0).string_value();

	  const swig_type_info *next_base = find_base(subname, base);
	  if (!next_base || skip >= (int) ops.size() || ops[skip] != '.')
	    break;
	  base = next_base;
	}

	member_value_pair tmp, *m = &tmp;
	if (!base || !(m->first = find_member(base, subname)))
	  m = find_member(subname, false);
	if (!m) {
	  error("member not found");
	  return octave_value_list();
	}

	octave_value_list args;
	if (!always_static &&
	    (!m->first || (!m->first->is_static() && !m->first->is_global())))
	  args.append(as_value());
	if (skip < (int) ops.size() && ops[skip] == '(' && 
	    ((m->first && m->first->method) || m->second.is_function() || 
	     m->second.is_function_handle())) {
	  args.append(*idx_it++);
	  ++skip;
	  sub_ovl = member_invoke(m, args, nargout);
	} else {
	  sub_ovl = member_deref(m, args);
	}
      }
      // index operator
      else {
	if (ops[skip] == '(' || ops[skip] == '{') {
	  const char *op_name = ops[skip] == '(' ? "__paren__" : "__brace__";
	  octave_value_list args;
	  args.append(*idx_it++);
	  ++skip;
	  if (!dispatch_index_op(op_name, args, sub_ovl)) {
	    error("error evaluating index operator");
	    return octave_value_list();
	  }
	} else {
	  error("unsupported subsref");
	  return octave_value_list();
	}
      }

      if (skip >= (int) ops.size())
	return sub_ovl;
      if (sub_ovl.length() < 1) {
	error("bad subs ref");
	return octave_value_list();
      }
      return sub_ovl(0).next_subsref(nargout, ops, idx, skip);
    }

    octave_value subsasgn(const std::string &ops, const std::list < octave_value_list > &idx, const octave_value &rhs) {
      assert(ops.size() > 0);
      assert(ops.size() == idx.size());

      std::list < octave_value_list >::const_iterator idx_it = idx.begin();
      int skip = 0;

      if (ops.size() > 1) {
	std::list < octave_value_list >::const_iterator last = idx.end();
	--last;
	std::list < octave_value_list > next_idx(idx.begin(), last);
	octave_value next_ov = subsref(ops.substr(0, ops.size() - 1), next_idx);
	next_ov.subsasgn(ops.substr(ops.size() - 1), std::list < octave_value_list > (1, *last), rhs);
      }

      else if (ops[skip] == '(' || ops[skip] == '{') {
	const char *op_name = ops[skip] == '(' ? "__paren_asgn__" : "__brace_asgn__";
	member_value_pair *m = find_member(op_name, false);
	if (m) {
	  octave_value_list args;
	  args.append(as_value());
	  args.append(*idx_it);
	  args.append(rhs);
	  member_invoke(m, args, 1);
	} else
	  error("%s member not found", op_name);
      }

      else if (ops[skip] == '.') {
	octave_value_list subname_ovl(*idx_it++);
	++skip;
	assert(subname_ovl.length() == 1 &&subname_ovl(0).is_string());
	std::string subname = subname_ovl(0).string_value();

	member_value_pair *m = find_member(subname, true);
	if (!m->first || !m->first->set_method) {
	  m->first = 0;
	  m->second = rhs;
	} else if (m->first->set_method) {
	  octave_value_list args;
	  if (!m->first->is_static() && !m->first->is_global())
	    args.append(as_value());
	  args.append(rhs);
	  m->first->set_method(args, 1);
	} else
	  error("member not assignable");
      } else
	error("unsupported subsasgn");

      return as_value();
    }

    virtual bool is_object() const {
      return true;
    }

    virtual bool is_string() const {
      octave_swig_type *nc_this = const_cast < octave_swig_type *>(this);
      return !!nc_this->find_member("__str__", false);
    }

    virtual std::string string_value(bool force = false) const {
      octave_value ret;
      if (!dispatch_unary_op("__str__", ret)) {
        error("__str__ method not defined");
        return std::string();
      }
      if (!ret.is_string()) {
	error("__str__ method did not return a string");
	return std::string();
      }
      return ret.string_value();
    }

    virtual double scalar_value(bool frc_str_conv = false) const {
      octave_value ret;
      if (!dispatch_unary_op("__float__", ret)) {
        error("__float__ method not defined");
      }
      return ret.scalar_value();
    }

#if SWIG_OCTAVE_PREREQ(4,2,0)
    virtual octave_value as_double(void) const {
      octave_value ret;
      if (!dispatch_unary_op("__float__", ret)) {
        error("__float__ method not defined");
      }
      return ret.as_double();
    }

    virtual octave_value as_single(void) const {
      octave_value ret;
      if (!dispatch_unary_op("__float__", ret)) {
        error("__float__ method not defined");
      }
      return ret.as_single();
    }
#endif

#if SWIG_OCTAVE_PREREQ(3,8,0)
    virtual octave_value map(octave_base_value::unary_mapper_t umap) const {
      const std::string opname = std::string("__") + octave_base_value::get_umap_name(umap) + std::string("__");
      octave_value ret;
      if (!dispatch_unary_op(opname, ret)) {
        error("%s", (opname + std::string(" method not found")).c_str());
        return octave_value();
      }
      return ret;
    }
#endif

#if SWIG_OCTAVE_PREREQ(3,3,52)
    virtual octave_map map_value() const {
      return octave_map();
    }
#else
    virtual Octave_map map_value() const {
      return Octave_map();
    }
#endif

    virtual string_vector map_keys() const {
      member_map tmp;
      load_members(tmp);

      string_vector keys(tmp.size());
      int k = 0;
      for (member_map::iterator it = tmp.begin(); it != tmp.end(); ++it)
	keys(k++) = it->first;

      return keys;
    }

    virtual bool save_ascii (std::ostream& os) {
      return true;
    }

    virtual bool load_ascii (std::istream& is) {
      return true;
    }

    virtual bool save_binary (std::ostream& os, bool& save_as_floats) {
      return true;
    }

    virtual bool load_binary (std::istream& is, bool swap, 
			      oct_mach_info::float_format fmt) {
      return true;
    }

#if defined (HAVE_HDF5)
# if SWIG_OCTAVE_PREREQ(4,0,0)
    virtual bool
      save_hdf5 (octave_hdf5_id loc_id, const char *name, bool save_as_floats) {
      return true;
    }

    virtual bool
      load_hdf5 (octave_hdf5_id loc_id, const char *name, bool have_h5giterate_bug) {
      return true;
    }
# else
    virtual bool
      save_hdf5 (hid_t loc_id, const char *name, bool save_as_floats) {
      return true;
    }

    virtual bool
      load_hdf5 (hid_t loc_id, const char *name, bool have_h5giterate_bug) {
      return true;
    }
# endif
#endif

    virtual octave_value convert_to_str(bool pad = false, bool force = false, char type = '"') const {
      return string_value();
    }

    virtual octave_value convert_to_str_internal(bool pad, bool force, char type) const {
      return string_value();
    }

    static bool dispatch_global_op(const std::string &symbol, const octave_value_list &args, octave_value &ret) {
      // we assume that SWIG_op_prefix-prefixed functions are installed in global namespace
      // (rather than any module namespace).

      octave_function *fcn = is_valid_function(symbol, std::string(), false);
      if (!fcn)
	return false;
#if SWIG_OCTAVE_PREREQ(4,4,0)
      octave::tree_evaluator& tw = octave::interpreter::the_interpreter()->get_evaluator();
      octave_value_list retval = fcn->call(tw, 1, args);
      if (retval.length() == 1)
        ret = retval(0);
#else
      ret = fcn->do_multi_index_op(1, args)(0);
#endif
      return true;
    }

    static octave_value dispatch_unary_op(const octave_base_value &x, const char *op_name) {
      octave_swig_type *ost = Swig::swig_value_deref(x);
      assert(ost);

      octave_value ret;
      if (ost->dispatch_unary_op(std::string("__") + op_name + std::string("__"), ret))
	return ret;
      std::string symbol = SWIG_op_prefix + ost->swig_type_name() + "_" + op_name;
      octave_value_list args;
      args.append(make_value_hack(x));
      if (dispatch_global_op(symbol, args, ret))
	return ret;

      error("could not dispatch unary operator");
      return octave_value();
    }

    static octave_value dispatch_binary_op(const octave_base_value &lhs, const octave_base_value &rhs, const char *op_name) {
      octave_swig_type *lhs_ost = Swig::swig_value_deref(lhs);
      octave_swig_type *rhs_ost = Swig::swig_value_deref(rhs);

      octave_value ret;
      if (lhs_ost && lhs_ost->dispatch_binary_op(std::string("__") + op_name + std::string("__"), rhs, ret))
	return ret;
      if (rhs_ost) {
        if (strlen(op_name) == 2  && (op_name[1] == 't' || op_name[1] == 'e')) {
          if (op_name[0] == 'l' && rhs_ost->dispatch_binary_op(std::string("__g") + op_name[1] + std::string("__"), lhs, ret))
            return ret;
          if (op_name[0] == 'g' && rhs_ost->dispatch_binary_op(std::string("__l") + op_name[1] + std::string("__"), lhs, ret))
            return ret;
        }
        if (rhs_ost->dispatch_binary_op(std::string("__r") + op_name + std::string("__"), lhs, ret))
          return ret;
      }

      std::string symbol;
      octave_value_list args;
      args.append(make_value_hack(lhs));
      args.append(make_value_hack(rhs));

      symbol = SWIG_op_prefix;
      symbol += lhs_ost ? lhs_ost->swig_type_name() : lhs.type_name();
      symbol += "_";
      symbol += op_name;
      symbol += "_";
      symbol += rhs_ost ? rhs_ost->swig_type_name() : rhs.type_name();
      if (dispatch_global_op(symbol, args, ret))
	return ret;

      symbol = SWIG_op_prefix;
      symbol += lhs_ost ? lhs_ost->swig_type_name() : lhs.type_name();
      symbol += "_";
      symbol += op_name;
      symbol += "_";
      symbol += "any";
      if (dispatch_global_op(symbol, args, ret))
	return ret;

      symbol = SWIG_op_prefix;
      symbol += "any";
      symbol += "_";
      symbol += op_name;
      symbol += "_";
      symbol += rhs_ost ? rhs_ost->swig_type_name() : rhs.type_name();
      if (dispatch_global_op(symbol, args, ret))
	return ret;

      error("could not dispatch binary operator");
      return octave_value();
    }

#if SWIG_OCTAVE_PREREQ(4,0,0)
    void print(std::ostream &os, bool pr_as_read_syntax = false)
#else
    void print(std::ostream &os, bool pr_as_read_syntax = false) const
#endif
    {
      if (is_string()) {
	os << string_value();
	return;
      }

      member_map tmp;
      load_members(tmp);

      indent(os);
      os << "{"; newline(os);
      increment_indent_level();
      for (unsigned int j = 0; j < types.size(); ++j) {
        indent(os);
	if (types[j].first->clientdata) {
	  const swig_octave_class *c = (const swig_octave_class *) types[j].first->clientdata;
	  os << c->name << ", ptr = " << types[j].second.ptr; newline(os);
	} else {
	  os << types[j].first->name << ", ptr = " << types[j].second.ptr; newline(os);
	}
      }
      for (member_map::const_iterator it = tmp.begin(); it != tmp.end(); ++it) {
        indent(os);
	if (it->second.first) {
	  const char *objtype = it->second.first->method ? "method" : "variable";
	  const char *modifier = (it->second.first->flags &1) ? "static " : (it->second.first->flags &2) ? "global " : "";
	  os << it->second.first->name << " (" << modifier << objtype << ")"; newline(os);
	  assert(it->second.first->name == it->first);
	} else {
	  os << it->first; newline(os);
	}
      }
      decrement_indent_level();
      indent(os);
      os << "}"; newline(os);
    }
  };

  // Octave tries hard to preserve pass-by-value semantics. Eg, assignments
  // will call clone() via make_unique() if there is more than one outstanding 
  // reference to the lhs, and forces the clone's reference count to 1 
  // (so you can't just increment your own count and return this).
  //
  // One way to fix this (without modifying Octave) is to add a level of
  // indirection such that clone copies ref-counted pointer and we keep 
  // pass-by-ref semantics (which are more natural/expected for C++ bindings).
  //
  // Supporting both pass-by-{ref,value} and toggling via %feature/option 
  // might be nice.

  class octave_swig_ref:public octave_base_value {
    octave_swig_type *ptr;
  public:
    octave_swig_ref(octave_swig_type *_ptr = 0)
      :ptr(_ptr)
      {
        // Ensure type_id() is set correctly
        if (t_id == -1) {
          t_id = octave_swig_ref::static_type_id();
        }
      }

    ~octave_swig_ref()
      { if (ptr) ptr->decref(); }

    octave_swig_type *get_ptr() const
      { return ptr; }

    octave_base_value *clone() const
      { if (ptr) ptr->incref(); return new octave_swig_ref(ptr); }

    octave_base_value *empty_clone() const
      { return new octave_swig_ref(0); }

    dim_vector dims(void) const 
      { return ptr->dims(); }

    bool is_defined() const
      { return ptr->is_defined(); }

    virtual bool is_map() const 
      { return ptr->is_map(); }

    virtual octave_value subsref(const std::string &ops, const std::list < octave_value_list > &idx) 
      { return ptr->subsref(ops, idx); }

    virtual octave_value_list subsref(const std::string &ops, const std::list < octave_value_list > &idx, int nargout)
      { return ptr->subsref(ops, idx, nargout); }

    octave_value subsasgn(const std::string &ops, const std::list < octave_value_list > &idx, const octave_value &rhs)
      { return ptr->subsasgn(ops, idx, rhs); }

    virtual bool is_object() const 
      { return ptr->is_object(); }

    virtual bool is_string() const 
      { return ptr->is_string(); }

    virtual std::string string_value(bool force = false) const 
      { return ptr->string_value(force); }

    virtual double scalar_value(bool frc_str_conv = false) const
      { return ptr->scalar_value(frc_str_conv); }

#if SWIG_OCTAVE_PREREQ(4,2,0)
    virtual octave_value as_double(void) const
      { return ptr->as_double(); }

    virtual octave_value as_single(void) const
      { return ptr->as_single(); }
#endif

#if SWIG_OCTAVE_PREREQ(3,8,0)
    virtual octave_value map(octave_base_value::unary_mapper_t umap) const
      { return ptr->map(umap); }
#endif

#if SWIG_OCTAVE_PREREQ(3,3,52)
    virtual octave_map map_value() const
      { return ptr->map_value(); }
#else
    virtual Octave_map map_value() const
      { return ptr->map_value(); }
#endif

    virtual string_vector map_keys() const
      { return ptr->map_keys(); }

    virtual bool save_ascii (std::ostream& os)
      { return ptr->save_ascii(os); }

    virtual bool load_ascii (std::istream& is)
      { return ptr->load_ascii(is); }

    virtual bool save_binary (std::ostream& os, bool& save_as_floats)
      { return ptr->save_binary(os, save_as_floats); }

    virtual bool load_binary (std::istream& is, bool swap, 
			      oct_mach_info::float_format fmt)
      { return ptr->load_binary(is, swap, fmt); }

#if defined (HAVE_HDF5)
# if SWIG_OCTAVE_PREREQ(4,0,0)
    virtual bool
      save_hdf5 (octave_hdf5_id loc_id, const char *name, bool save_as_floats)
      { return ptr->save_hdf5(loc_id, name, save_as_floats); }

    virtual bool
      load_hdf5 (octave_hdf5_id loc_id, const char *name, bool have_h5giterate_bug)
      { return ptr->load_hdf5(loc_id, name, have_h5giterate_bug); }
# else
    virtual bool
      save_hdf5 (hid_t loc_id, const char *name, bool save_as_floats)
      { return ptr->save_hdf5(loc_id, name, save_as_floats); }

    virtual bool
      load_hdf5 (hid_t loc_id, const char *name, bool have_h5giterate_bug)
      { return ptr->load_hdf5(loc_id, name, have_h5giterate_bug); }
# endif
#endif

    virtual octave_value convert_to_str(bool pad = false, bool force = false, char type = '"') const
      { return ptr->convert_to_str(pad, force, type); }

    virtual octave_value convert_to_str_internal(bool pad, bool force, char type) const
      { return ptr->convert_to_str_internal(pad, force, type); }

#if SWIG_OCTAVE_PREREQ(4,0,0)
    void print(std::ostream &os, bool pr_as_read_syntax = false)
#else
    void print(std::ostream &os, bool pr_as_read_syntax = false) const
#endif
      { return ptr->print(os, pr_as_read_syntax); }

    virtual type_conv_info numeric_conversion_function(void) const {
      return octave_base_value::type_conv_info (default_numeric_conversion_function,
                                                octave_scalar::static_type_id ());
    }

  private:
    static octave_base_value *default_numeric_conversion_function (const octave_base_value& a) {
      const octave_swig_ref& v = dynamic_cast<const octave_swig_ref&>(a);
      return new octave_scalar(v.scalar_value());
    }

#if !SWIG_OCTAVE_PREREQ(4,0,0)
    DECLARE_OCTAVE_ALLOCATOR;
#endif
    DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA;
  };
#if !SWIG_OCTAVE_PREREQ(4,0,0)
  DEFINE_OCTAVE_ALLOCATOR(octave_swig_ref);
#endif
  DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA(octave_swig_ref, "swig_ref", "swig_ref");

  class octave_swig_packed:public octave_base_value {
    swig_type_info *type;
    std::vector < char > buf;
  public:

    octave_swig_packed(swig_type_info *_type = 0, const void *_buf = 0, size_t _buf_len = 0)
      :	type(_type), buf((const char*)_buf, (const char*)_buf + _buf_len)
      {
        // Ensure type_id() is set correctly
        if (t_id == -1) {
          t_id = octave_swig_packed::static_type_id();
        }
      }

    bool copy(swig_type_info *outtype, void *ptr, size_t sz) const {
      if (outtype && outtype != type)
	return false;
      assert(sz <= buf.size());
      std::copy(buf.begin(), buf.begin()+sz, (char*)ptr);
      return true;
    }

    octave_base_value *clone() const {
      return new octave_swig_packed(*this);
    }

    octave_base_value *empty_clone() const {
      return new octave_swig_packed();
    }

    bool is_defined() const {
      return true;
    }

#if SWIG_OCTAVE_PREREQ(4,0,0)
    void print(std::ostream &os, bool pr_as_read_syntax = false)
#else
    void print(std::ostream &os, bool pr_as_read_syntax = false) const
#endif
    {
      indent(os);
      os << "swig packed type: name = " << (type ? type->name : std::string()) << ", len = " << buf.size(); newline(os);
    }


    virtual bool save_ascii (std::ostream& os) {
      return true;
    }

    virtual bool load_ascii (std::istream& is) {
      return true;
    }

    virtual bool save_binary (std::ostream& os, bool& save_as_floats) {
      return true;
    }

    virtual bool load_binary (std::istream& is, bool swap, 
			      oct_mach_info::float_format fmt) {
      return true;
    }

#if defined (HAVE_HDF5)
# if SWIG_OCTAVE_PREREQ(4,0,0)
    virtual bool
      save_hdf5 (octave_hdf5_id loc_id, const char *name, bool save_as_floats) {
      return true;
    }

    virtual bool
      load_hdf5 (octave_hdf5_id loc_id, const char *name, bool have_h5giterate_bug) {
      return true;
    }
# else
    virtual bool
      save_hdf5 (hid_t loc_id, const char *name, bool save_as_floats) {
      return true;
    }

    virtual bool
      load_hdf5 (hid_t loc_id, const char *name, bool have_h5giterate_bug) {
      return true;
    }
# endif
#endif

  private:
#if !SWIG_OCTAVE_PREREQ(4,0,0)
    DECLARE_OCTAVE_ALLOCATOR;
#endif
    DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA;
  };
#if !SWIG_OCTAVE_PREREQ(4,0,0)
  DEFINE_OCTAVE_ALLOCATOR(octave_swig_packed);
#endif
  DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA(octave_swig_packed, "swig_packed", "swig_packed");

  SWIGRUNTIME octave_value_list octave_set_immutable(const octave_value_list &args, int nargout) {
    error("attempt to set immutable member variable");
    return octave_value_list();
  }

  struct octave_value_ref {
    const octave_value_list &ovl;
    int j;

    octave_value_ref(const octave_value_list &_ovl, int _j)
      :ovl(_ovl), j(_j) { }

    operator  octave_value() const {
      return ovl(j);
    }

    octave_value operator*() const {
      return ovl(j);
    }
  };


namespace Swig {

  SWIGRUNTIME octave_base_value *swig_value_ref(octave_swig_type *ost) {
    return new octave_swig_ref(ost);
  }

  SWIGRUNTIME octave_swig_type *swig_value_deref(octave_value ov) {
    if (
#if SWIG_OCTAVE_PREREQ(4,4,0)
      ov.iscell()
#else
      ov.is_cell()
#endif
      && ov.rows() == 1 && ov.columns() == 1)
      ov = ov.cell_value()(0);
    return swig_value_deref(*ov.internal_rep());
  }

  SWIGRUNTIME octave_swig_type *swig_value_deref(const octave_base_value &ov) {
    if (ov.type_id() != octave_swig_ref::static_type_id())
      return 0;
    const octave_swig_ref *osr = static_cast < const octave_swig_ref *>(&ov);
    return osr->get_ptr();
  }

}


#define swig_unary_op(name) \
SWIGRUNTIME octave_value swig_unary_op_##name(const octave_base_value &x) { \
  return octave_swig_type::dispatch_unary_op(x,#name); \
}
#define swig_binary_op(name) \
SWIGRUNTIME octave_value swig_binary_op_##name(const octave_base_value&lhs,const octave_base_value &rhs) { \
  return octave_swig_type::dispatch_binary_op(lhs,rhs,#name); \
}
#if SWIG_OCTAVE_PREREQ(4,4,0)
#define swigreg_unary_op(name) \
if (!octave_value_typeinfo::lookup_unary_op(octave_value::op_##name,tid)) \
typeinfo.register_unary_op(octave_value::op_##name,tid,swig_unary_op_##name);
#else
#define swigreg_unary_op(name) \
if (!octave_value_typeinfo::lookup_unary_op(octave_value::op_##name,tid)) \
octave_value_typeinfo::register_unary_op(octave_value::op_##name,tid,swig_unary_op_##name);
#endif
#if SWIG_OCTAVE_PREREQ(4,4,0)
#define swigreg_binary_op(name) \
if (!octave_value_typeinfo::lookup_binary_op(octave_value::op_##name,tid1,tid2)) \
typeinfo.register_binary_op(octave_value::op_##name,tid1,tid2,swig_binary_op_##name);
#else
#define swigreg_binary_op(name) \
if (!octave_value_typeinfo::lookup_binary_op(octave_value::op_##name,tid1,tid2)) \
octave_value_typeinfo::register_binary_op(octave_value::op_##name,tid1,tid2,swig_binary_op_##name);
#endif

  swig_unary_op(not);
  swig_unary_op(uplus);
  swig_unary_op(uminus);
  swig_unary_op(transpose);
  swig_unary_op(hermitian);
  swig_unary_op(incr);
  swig_unary_op(decr);

  swig_binary_op(add);
  swig_binary_op(sub);
  swig_binary_op(mul);
  swig_binary_op(div);
  swig_binary_op(pow);
  swig_binary_op(ldiv);
#if !SWIG_OCTAVE_PREREQ(4,2,0)
  swig_binary_op(lshift);
  swig_binary_op(rshift);
#endif
  swig_binary_op(lt);
  swig_binary_op(le);
  swig_binary_op(eq);
  swig_binary_op(ge);
  swig_binary_op(gt);
  swig_binary_op(ne);
  swig_binary_op(el_mul);
  swig_binary_op(el_div);
  swig_binary_op(el_pow);
  swig_binary_op(el_ldiv);
  swig_binary_op(el_and);
  swig_binary_op(el_or);

  SWIGRUNTIME void SWIG_InstallUnaryOps(int tid) {
#if SWIG_OCTAVE_PREREQ(4,4,0)
    octave::type_info& typeinfo = octave::interpreter::the_interpreter()->get_type_info();
#endif
    swigreg_unary_op(not);
    swigreg_unary_op(uplus);
    swigreg_unary_op(uminus);
    swigreg_unary_op(transpose);
    swigreg_unary_op(hermitian);
    swigreg_unary_op(incr);
    swigreg_unary_op(decr);
  }
  SWIGRUNTIME void SWIG_InstallBinaryOps(int tid1, int tid2) {
#if SWIG_OCTAVE_PREREQ(4,4,0)
    octave::type_info& typeinfo = octave::interpreter::the_interpreter()->get_type_info();
#endif
    swigreg_binary_op(add);
    swigreg_binary_op(sub);
    swigreg_binary_op(mul);
    swigreg_binary_op(div);
    swigreg_binary_op(pow);
    swigreg_binary_op(ldiv);
#if !SWIG_OCTAVE_PREREQ(4,2,0)
    swigreg_binary_op(lshift);
    swigreg_binary_op(rshift);
#endif
    swigreg_binary_op(lt);
    swigreg_binary_op(le);
    swigreg_binary_op(eq);
    swigreg_binary_op(ge);
    swigreg_binary_op(gt);
    swigreg_binary_op(ne);
    swigreg_binary_op(el_mul);
    swigreg_binary_op(el_div);
    swigreg_binary_op(el_pow);
    swigreg_binary_op(el_ldiv);
    swigreg_binary_op(el_and);
    swigreg_binary_op(el_or);
  }
  SWIGRUNTIME void SWIG_InstallOps(int tid) {
    // here we assume that tid are conseq integers increasing from zero, and 
    // that our tid is the last one. might be better to have explicit string 
    // list of types we should bind to, and use lookup_type to resolve their tid.

    SWIG_InstallUnaryOps(tid);
    SWIG_InstallBinaryOps(tid, tid);
    for (int j = 0; j < tid; ++j) {
      SWIG_InstallBinaryOps(j, tid);
      SWIG_InstallBinaryOps(tid, j);
    }
  }

SWIGRUNTIME octave_value SWIG_Octave_NewPointerObj(void *ptr, swig_type_info *type, int flags) {
  int own = (flags &SWIG_POINTER_OWN) ? SWIG_POINTER_OWN : 0;

#ifdef SWIG_DIRECTORS
  Swig::Director *d = Swig::get_rtdir(ptr);
  if (d && Swig::swig_director_get_self(d))
    return Swig::swig_director_get_self(d)->as_value();
#endif
  return Swig::swig_value_ref(new octave_swig_type(ptr, type, own));
}

SWIGRUNTIME int SWIG_Octave_ConvertPtrAndOwn(octave_value ov, void **ptr, swig_type_info *type, int flags, int *own) {
    if (
#if SWIG_OCTAVE_PREREQ(4,4,0)
      ov.iscell()
#else
      ov.is_cell()
#endif
      && ov.rows() == 1 && ov.columns() == 1)
    ov = ov.cell_value()(0);
  if (!ov.is_defined() ||
      (ov.is_matrix_type() && ov.rows() == 0 && ov.columns() == 0) ) {
    if (ptr)
      *ptr = 0;
    return (flags & SWIG_POINTER_NO_NULL) ? SWIG_NullReferenceError : SWIG_OK;
  }
  if (ov.type_id() != octave_swig_ref::static_type_id())
    return SWIG_ERROR;
  octave_swig_ref *osr = static_cast < octave_swig_ref *>(ov.internal_rep());
  octave_swig_type *ost = osr->get_ptr();
  return ost->cast(ptr, type, own, flags);
}

SWIGRUNTIME octave_value SWIG_Octave_NewPackedObj(void *ptr, size_t sz, swig_type_info *type) {
  return new octave_swig_packed(type, (char *) ptr, sz);
}

SWIGRUNTIME int SWIG_Octave_ConvertPacked(const octave_value &ov, void *ptr, size_t sz, swig_type_info *type) {
  if (!ov.is_defined())
    return SWIG_ERROR;
  if (ov.type_id() != octave_swig_packed::static_type_id())
    return SWIG_ERROR;
  octave_swig_packed *ost = static_cast < octave_swig_packed *>(ov.internal_rep());
  return ost->copy(type, (char *) ptr, sz) ? SWIG_OK : SWIG_ERROR;
}

SWIGRUNTIMEINLINE void SWIG_Octave_SetConstant(octave_swig_type *module_ns, const std::string &name, const octave_value &ov) {
  module_ns->assign(name, ov);
}

SWIGRUNTIMEINLINE octave_value SWIG_Octave_GetGlobalValue(std::string name) {
#if SWIG_OCTAVE_PREREQ(4,4,0)
  octave::symbol_table& symtab = octave::interpreter::the_interpreter()->get_symbol_table();
  return symtab.global_varval(name);
#else
  return get_global_value(name, true);
#endif
}

SWIGRUNTIME void SWIG_Octave_SetGlobalValue(std::string name, const octave_value& value) {
#if SWIG_OCTAVE_PREREQ(4,4,0)
  octave::symbol_table& symtab = octave::interpreter::the_interpreter()->get_symbol_table();
  symtab.global_assign(name, value);
#else
  set_global_value(name, value);
#endif
}

SWIGRUNTIME void SWIG_Octave_LinkGlobalValue(std::string name) {
#if SWIG_OCTAVE_PREREQ(4,4,0)
  octave::symbol_table& symtab = octave::interpreter::the_interpreter()->get_symbol_table();
  octave::symbol_scope symscope = octave::interpreter::the_interpreter()->get_current_scope();
  symscope.assign(name, symtab.global_varval(name));
  symscope.mark_global(name);
#else
#if !SWIG_OCTAVE_PREREQ(3,2,0)
  link_to_global_variable(curr_sym_tab->lookup(name, true));
#else
#if !SWIG_OCTAVE_PREREQ(3,8,0)
  symbol_table::varref(name);
#endif
  symbol_table::mark_global(name);
#endif
#endif
}

SWIGRUNTIME swig_module_info *SWIG_Octave_GetModule(void *clientdata) {
  octave_value ov = SWIG_Octave_GetGlobalValue("__SWIG_MODULE__" SWIG_TYPE_TABLE_NAME SWIG_RUNTIME_VERSION);
  if (!ov.is_defined() ||
      ov.type_id() != octave_swig_packed::static_type_id())
    return 0;
  const octave_swig_packed* osp = 
    static_cast < const octave_swig_packed *> (ov.internal_rep());
  swig_module_info *pointer = 0;
  osp->copy(0, &pointer, sizeof(swig_module_info *));
  return pointer;
}

SWIGRUNTIME void SWIG_Octave_SetModule(void *clientdata, swig_module_info *pointer) {
  octave_value ov = new octave_swig_packed(0, &pointer, sizeof(swig_module_info *));
  SWIG_Octave_SetGlobalValue("__SWIG_MODULE__" SWIG_TYPE_TABLE_NAME SWIG_RUNTIME_VERSION, ov);
}
