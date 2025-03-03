#ifdef __cplusplus

/*
  GC_VALUE is used as a replacement of Ruby's VALUE.
  GC_VALUE automatically handles registering and unregistering
  of the underlying Ruby object with the GC.

  It can be used if you want to create STL containers of VALUEs, such as:
  
     std::vector< GC_VALUE >;

  or as a member variable:
  
     struct A {
       GC_VALUE _obj;
       A(VALUE o) : _obj(o) {
       }
     };

   or as a input/output value (not much use for this, as VALUE works just as
   well here, thou):

     GC_VALUE func(GC_VALUE obj) { 
       GC_VALUE out = rb_obj_classname(obj);
       return out;
     }


   GC_VALUE is 'visible' at the wrapped side, so you can do:

      %template(RubyVector) std::vector<swig::GC_VALUE>;

   and all the proper typemaps will be used.
   
*/

%fragment("GC_VALUE_definition","header") {
namespace swig {
  class SwigGCReferences {
    VALUE _hash;

    SwigGCReferences() : _hash(Qnil) {
    }
    ~SwigGCReferences() {
      if (_hash != Qnil)
        rb_gc_unregister_address(&_hash);
    }
    static void EndProcHandler(VALUE) {
      // Ruby interpreter ending - _hash can no longer be accessed.
      SwigGCReferences &s_references = instance();
      s_references._hash = Qnil;
    }
  public:
    static SwigGCReferences& instance() {
      // Hash of all GC_VALUE's currently in use
      static SwigGCReferences s_references;

      return s_references;
    }
    static void initialize() {
      SwigGCReferences &s_references = instance();
      if (s_references._hash == Qnil) {
        rb_set_end_proc(&EndProcHandler, Qnil);
        s_references._hash = rb_hash_new();
        rb_gc_register_address(&s_references._hash);
      }
    }
    void GC_register(VALUE& obj) {
      if (FIXNUM_P(obj) || SPECIAL_CONST_P(obj) || SYMBOL_P(obj))
        return;
      if (_hash != Qnil) {
        VALUE val = rb_hash_aref(_hash, obj);
        unsigned n = FIXNUM_P(val) ? NUM2UINT(val) : 0;
        ++n;
        rb_hash_aset(_hash, obj, INT2NUM(n));
      }
    }
    void GC_unregister(const VALUE& obj) {
      if (FIXNUM_P(obj) || SPECIAL_CONST_P(obj) || SYMBOL_P(obj))
        return;
      // this test should not be needed but I've noticed some very erratic
      // behavior of none being unregistered in some very rare situations.
      if (BUILTIN_TYPE(obj) == T_NONE)
        return;
      if (_hash != Qnil) {
        VALUE val = rb_hash_aref(_hash, obj);
        unsigned n = FIXNUM_P(val) ? NUM2UINT(val) : 1;
        --n;
        if (n)
          rb_hash_aset(_hash, obj, INT2NUM(n));
        else
          rb_hash_delete(_hash, obj);
      }
    }
  };

  class GC_VALUE {
  protected:
    VALUE  _obj;

    static ID hash_id;
    static ID   lt_id;
    static ID   gt_id;
    static ID   eq_id;
    static ID   le_id;
    static ID   ge_id;

    static ID  pos_id;
    static ID  neg_id;
    static ID  inv_id;

    static ID  add_id;
    static ID  sub_id;
    static ID  mul_id;
    static ID  div_id;
    static ID  mod_id;

    static ID  and_id;
    static ID   or_id;
    static ID  xor_id;

    static ID  lshift_id;
    static ID  rshift_id;

    struct OpArgs
    {
      VALUE src;
      ID    id;
      int   nargs;
      VALUE target;
    };


  public:
    GC_VALUE() : _obj(Qnil)
    {
    }

    GC_VALUE(const GC_VALUE& item) : _obj(item._obj)
    {
      SwigGCReferences::instance().GC_register(_obj);
    }
    
    GC_VALUE(VALUE obj) :_obj(obj)
    {
      SwigGCReferences::instance().GC_register(_obj);
    }
    
    ~GC_VALUE() 
    {
      SwigGCReferences::instance().GC_unregister(_obj);
    }
    
    GC_VALUE & operator=(const GC_VALUE& item) 
    {
      SwigGCReferences::instance().GC_unregister(_obj);
      _obj = item._obj;
      SwigGCReferences::instance().GC_register(_obj);
      return *this;
    }

    operator VALUE() const
    {
      return _obj;
    }

    VALUE inspect() const
    {
      return rb_inspect(_obj);
    }

    VALUE to_s() const
    {
      return rb_inspect(_obj);
    }

    static VALUE swig_rescue_swallow(VALUE, VALUE)
    {
      /*
      VALUE errstr = rb_obj_as_string(rb_errinfo());
      printf("Swallowing error: '%s'\n", RSTRING_PTR(StringValue(errstr)));
      */
      return Qnil; /* Swallow Ruby exception */
    }

    static VALUE swig_rescue_funcall(VALUE p)
    {
      OpArgs* args = (OpArgs*) p;
      return rb_funcall(args->src, args->id, args->nargs, args->target);
    }

    bool relational_equal_op(const GC_VALUE& other, const ID& op_id, bool (*op_func)(const VALUE& a, const VALUE& b)) const
    {
      if (FIXNUM_P(_obj) && FIXNUM_P(other._obj)) {
        return op_func(_obj, other._obj);
      }
      bool res = false;
      VALUE ret = Qnil;
      SWIG_RUBY_THREAD_BEGIN_BLOCK;
      if (rb_respond_to(_obj, op_id)) {
        OpArgs  args;
        args.src    = _obj;
        args.id     = op_id;
        args.nargs  = 1;
        args.target = VALUE(other);
        ret = rb_rescue(VALUEFUNC(swig_rescue_funcall), VALUE(&args),
                       (VALUEFUNC(swig_rescue_swallow)), Qnil);
      }
      if (ret == Qnil) {
        VALUE a = rb_funcall(         _obj, hash_id, 0 );
        VALUE b = rb_funcall( VALUE(other), hash_id, 0 );
        res = op_func(a, b);
      } else {
        res = RTEST(ret);
      }
      SWIG_RUBY_THREAD_END_BLOCK;
      return res;
    }

    static bool operator_eq(const VALUE& a, const VALUE& b) { return a == b; }
    static bool operator_lt(const VALUE& a, const VALUE& b) { return a < b; }
    static bool operator_le(const VALUE& a, const VALUE& b) { return a <= b; }
    static bool operator_gt(const VALUE& a, const VALUE& b) { return a > b; }
    static bool operator_ge(const VALUE& a, const VALUE& b) { return a >= b; }

    bool operator==(const GC_VALUE& other) const { return relational_equal_op(other, eq_id, operator_eq); }
    bool operator<(const GC_VALUE& other) const { return relational_equal_op(other, lt_id, operator_lt); }
    bool operator<=(const GC_VALUE& other) const { return relational_equal_op(other, le_id, operator_le); }
    bool operator>(const GC_VALUE& other) const { return relational_equal_op(other, gt_id, operator_gt); }
    bool operator>=(const GC_VALUE& other) const { return relational_equal_op(other, ge_id, operator_ge); }

    bool operator!=(const GC_VALUE& other) const
    {
      return !(this->operator==(other));
    }

    GC_VALUE unary_op(const ID& op_id) const
    {
      VALUE ret = Qnil;
      SWIG_RUBY_THREAD_BEGIN_BLOCK;
      OpArgs  args;
      args.src    = _obj;
      args.id     = op_id;
      args.nargs  = 0;
      args.target = Qnil;
      ret = rb_rescue(VALUEFUNC(swig_rescue_funcall), VALUE(&args),
                     (VALUEFUNC(swig_rescue_swallow)), Qnil);
      SWIG_RUBY_THREAD_END_BLOCK;
      return ret;
    }

    GC_VALUE operator+() const { return unary_op(pos_id); }
    GC_VALUE operator-() const { return unary_op(neg_id); }
    GC_VALUE operator~() const { return unary_op(inv_id); }

    GC_VALUE binary_op(const GC_VALUE& other, const ID& op_id) const
    {
      VALUE ret = Qnil;
      SWIG_RUBY_THREAD_BEGIN_BLOCK;
      OpArgs  args;
      args.src    = _obj;
      args.id     = op_id;
      args.nargs  = 1;
      args.target = VALUE(other);
      ret = rb_rescue(VALUEFUNC(swig_rescue_funcall), VALUE(&args),
                     (VALUEFUNC(swig_rescue_swallow)), Qnil);
      SWIG_RUBY_THREAD_END_BLOCK;
      return GC_VALUE(ret);
    }

    GC_VALUE operator+(const GC_VALUE& other) const { return binary_op(other, add_id); }
    GC_VALUE operator-(const GC_VALUE& other) const { return binary_op(other, sub_id); }
    GC_VALUE operator*(const GC_VALUE& other) const { return binary_op(other, mul_id); }
    GC_VALUE operator/(const GC_VALUE& other) const { return binary_op(other, div_id); }
    GC_VALUE operator%(const GC_VALUE& other) const { return binary_op(other, mod_id); }
    GC_VALUE operator&(const GC_VALUE& other) const { return binary_op(other, and_id); }
    GC_VALUE operator^(const GC_VALUE& other) const { return binary_op(other, xor_id); }
    GC_VALUE operator|(const GC_VALUE& other) const { return binary_op(other, or_id); }
    GC_VALUE operator<<(const GC_VALUE& other) const { return binary_op(other, lshift_id); }
    GC_VALUE operator>>(const GC_VALUE& other) const { return binary_op(other, rshift_id); }
  };

  ID  GC_VALUE::hash_id = rb_intern("hash");
  ID  GC_VALUE::lt_id = rb_intern("<");
  ID  GC_VALUE::gt_id = rb_intern(">");
  ID  GC_VALUE::eq_id = rb_intern("==");
  ID  GC_VALUE::le_id = rb_intern("<=");
  ID  GC_VALUE::ge_id = rb_intern(">=");

  ID  GC_VALUE::pos_id = rb_intern("+@");
  ID  GC_VALUE::neg_id = rb_intern("-@");
  ID  GC_VALUE::inv_id = rb_intern("~");

  ID  GC_VALUE::add_id = rb_intern("+");
  ID  GC_VALUE::sub_id = rb_intern("-");
  ID  GC_VALUE::mul_id = rb_intern("*");
  ID  GC_VALUE::div_id = rb_intern("/");
  ID  GC_VALUE::mod_id = rb_intern("%");

  ID  GC_VALUE::and_id = rb_intern("&");
  ID  GC_VALUE::or_id  = rb_intern("|");
  ID  GC_VALUE::xor_id = rb_intern("^");

  ID  GC_VALUE::lshift_id = rb_intern("<<");
  ID  GC_VALUE::rshift_id = rb_intern(">>");

  typedef GC_VALUE LANGUAGE_OBJ;

} // namespace swig

} // %fragment(GC_VALUE_definition)



namespace swig {

  %apply VALUE   {GC_VALUE};

  // Make sure this is the last typecheck done
  %typecheck(999999,fragment="GC_VALUE_definition",noblock=1) GC_VALUE, GC_VALUE&, 
    const GC_VALUE& { $1 = 1; };

  /* For input */
  %typemap(in,fragment="GC_VALUE_definition",noblock=1) GC_VALUE* (GC_VALUE r), GC_VALUE& (GC_VALUE r)  {
     r = $input; $1 = &r;
   }

  /* For output */
  %typemap(out,fragment="GC_VALUE_definition",noblock=1)  GC_VALUE {
    $result = (VALUE)$1;
  }
  
  %typemap(out,fragment="GC_VALUE_definition",noblock=1)  GC_VALUE*, GC_VALUE const & {
    $result = (VALUE)*$1;
  }

  %nodirector GC_VALUE;

  // We ignore the constructor so that user can never create a GC_VALUE 
  // manually
  %ignore GC_VALUE::GC_VALUE;

  struct GC_VALUE {
    VALUE inspect() const;
    VALUE to_s() const;
    GC_VALUE();
  protected:
    GC_VALUE(const GC_VALUE&);
    ~GC_VALUE();
  };

  %exception GC_VALUE {};


  %ignore LANGUAGE_OBJ;
  typedef GC_VALUE LANGUAGE_OBJ;
}


%init {
  swig::SwigGCReferences::initialize();
}



//
// Fragment that contains traits to properly deal with GC_VALUE.
// These functions may be invoked as a need of the from(), asval(),
// asptr() and as() template functors, usually used in %typemaps.
//
%fragment(SWIG_Traits_frag(swig::GC_VALUE),"header",fragment="StdTraits",fragment="GC_VALUE_definition") {
namespace swig {
  template <>  struct traits<GC_VALUE > {
    typedef value_category category;
    static const char* type_name() { return "GC_VALUE"; }
  };
  
  template <>  struct traits_from<GC_VALUE> {
    typedef GC_VALUE value_type;
    static VALUE from(const value_type& val) {
      return static_cast<VALUE>(val);
    }
  };
  
  template <> 
  struct traits_check<GC_VALUE, value_category> {
    static bool check(GC_VALUE) {
      return true;
    }
  };
  
  template <>  struct traits_asval<GC_VALUE > {   
    typedef GC_VALUE value_type;
    static int asval(VALUE obj, value_type *val) {
      if (val) *val = obj;
      return SWIG_OK;
    }
  };
} // swig
} // %fragment(traits for swig::GC_VALUE)


#endif  // __cplusplus

