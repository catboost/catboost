//
//   Maps
//
%fragment("StdMapCommonTraits","header",fragment="StdSequenceTraits")
{
  namespace swig {
    template <class ValueType>
    struct from_key_oper 
    {
      typedef const ValueType& argument_type;
      typedef  VALUE result_type;
      result_type operator()(argument_type v) const
      {
	return swig::from(v.first);
      }
    };

    template <class ValueType>
    struct from_value_oper 
    {
      typedef const ValueType& argument_type;
      typedef  VALUE result_type;
      result_type operator()(argument_type v) const
      {
	return swig::from(v.second);
      }
    };

    template<class OutIterator, class FromOper, 
	     class ValueType = typename OutIterator::value_type>
    struct MapIterator_T : ConstIteratorClosed_T<OutIterator, ValueType, FromOper>
    {
      MapIterator_T(OutIterator curr, OutIterator first, OutIterator last, VALUE seq)
	: ConstIteratorClosed_T<OutIterator,ValueType,FromOper>(curr, first, last, seq)
      {
      }
    };


    template<class OutIterator,
	     class FromOper = from_key_oper<typename OutIterator::value_type> >
    struct MapKeyIterator_T : MapIterator_T<OutIterator, FromOper>
    {
      MapKeyIterator_T(OutIterator curr, OutIterator first, OutIterator last, VALUE seq)
	: MapIterator_T<OutIterator, FromOper>(curr, first, last, seq)
      {
      }
    };

    template<typename OutIter>
    inline ConstIterator*
    make_output_key_iterator(const OutIter& current, const OutIter& begin, 
			     const OutIter& end, VALUE seq = 0)
    {
      return new MapKeyIterator_T<OutIter>(current, begin, end, seq);
    }

    template<class OutIterator,
	     class FromOper = from_value_oper<typename OutIterator::value_type> >
    struct MapValueIterator_T : MapIterator_T<OutIterator, FromOper>
    {
      MapValueIterator_T(OutIterator curr, OutIterator first, OutIterator last, VALUE seq)
	: MapIterator_T<OutIterator, FromOper>(curr, first, last, seq)
      {
      }
    };
    

    template<typename OutIter>
    inline ConstIterator*
    make_output_value_iterator(const OutIter& current, const OutIter& begin, 
			       const OutIter& end, VALUE seq = 0)
    {
      return new MapValueIterator_T<OutIter>(current, begin, end, seq);
    }
  }
}

%fragment("StdMapTraits","header",fragment="StdMapCommonTraits")
{
  namespace swig {
    template <class RubySeq, class K, class T >
    inline void
    assign(const RubySeq& rubyseq, std::map<K,T > *map) {
      typedef typename std::map<K,T>::value_type value_type;
      typename RubySeq::const_iterator it = rubyseq.begin();
      for (;it != rubyseq.end(); ++it) {
	map->insert(value_type(it->first, it->second));
      }
    }

    template <class K, class T>
    struct traits_asptr<std::map<K,T> >  {
      typedef std::map<K,T> map_type;
      static int asptr(VALUE obj, map_type **val) {
	int res = SWIG_ERROR;
	if ( TYPE(obj) == T_HASH ) {
	  static ID id_to_a = rb_intern("to_a");
	  VALUE items = rb_funcall(obj, id_to_a, 0);
	  res = traits_asptr_stdseq<std::map<K,T>, std::pair<K, T> >::asptr(items, val);
	} else {
	  map_type *p;
	  swig_type_info *descriptor = swig::type_info<map_type>();
	  res = descriptor ? SWIG_ConvertPtr(obj, (void **)&p, descriptor, 0) : SWIG_ERROR;
	  if (SWIG_IsOK(res) && val)  *val = p;
	}
	return res;
      }
    };
      
    template <class K, class T >
    struct traits_from<std::map<K,T> >  {
      typedef std::map<K,T> map_type;
      typedef typename map_type::const_iterator const_iterator;
      typedef typename map_type::size_type size_type;
            
      static VALUE from(const map_type& map) {
	swig_type_info *desc = swig::type_info<map_type>();
	if (desc && desc->clientdata) {
	  return SWIG_NewPointerObj(new map_type(map), desc, SWIG_POINTER_OWN);
	} else {
	  size_type size = map.size();
	  int rubysize = (size <= (size_type) INT_MAX) ? (int) size : -1;
	  if (rubysize < 0) {
	    SWIG_RUBY_THREAD_BEGIN_BLOCK;
	    rb_raise( rb_eRuntimeError, "map size not valid in Ruby");
	    SWIG_RUBY_THREAD_END_BLOCK;
	    return Qnil;
	  }
	  VALUE obj = rb_hash_new();
	  for (const_iterator i= map.begin(); i!= map.end(); ++i) {
	    VALUE key = swig::from(i->first);
	    VALUE val = swig::from(i->second);
	    rb_hash_aset(obj, key, val);
	  }
	  return obj;
	}
      }
    };
  }
}

%define %swig_map_common(Map...)
  %swig_container_methods(%arg(Map));
  // %swig_sequence_iterator(%arg(Map));

  %extend {
    
    VALUE __delete__(const key_type& key) {
      Map::iterator i = self->find(key);
      if (i != self->end()) {
	self->erase(i);
	return swig::from( key );
      }
      else {
	return Qnil;
      }
    }
    
    bool has_key(const key_type& key) const {
      Map::const_iterator i = self->find(key);
      return i != self->end();
    }
    
    VALUE keys() {
      Map::size_type size = self->size();
      int rubysize = (size <= (Map::size_type) INT_MAX) ? (int) size : -1;
      if (rubysize < 0) {
	SWIG_RUBY_THREAD_BEGIN_BLOCK;
	rb_raise(rb_eRuntimeError, "map size not valid in Ruby");
	SWIG_RUBY_THREAD_END_BLOCK;
	return Qnil;
      }
      VALUE ary = rb_ary_new2(rubysize);
      Map::const_iterator i = self->begin();
      Map::const_iterator e = self->end();
      for ( ; i != e; ++i ) {
	rb_ary_push( ary, swig::from(i->first) );
      }
      return ary;
    }

    Map* each()
      {
	if ( !rb_block_given_p() )
	  rb_raise( rb_eArgError, "no block given");

	VALUE k, v;
	Map::iterator i = self->begin();
	Map::iterator e = self->end();
	for ( ; i != e; ++i )
	  {
	    const Map::key_type&    key = i->first;
	    const Map::mapped_type& val = i->second;

	    k = swig::from<Map::key_type>(key);
	    v = swig::from<Map::mapped_type>(val);
	    rb_yield_values(2, k, v);
	  }
	
	return self;
      }

    %newobject select;
    Map* select() {
      if ( !rb_block_given_p() )
	rb_raise( rb_eArgError, "no block given" );

      Map* r = new Map;
      Map::iterator i = $self->begin();
      Map::iterator e = $self->end();
      for ( ; i != e; ++i )
	{
	  VALUE k = swig::from<Map::key_type>(i->first);
	  VALUE v = swig::from<Map::mapped_type>(i->second);
	  if ( RTEST( rb_yield_values(2, k, v) ) )
	    $self->insert(r->end(), *i);
	}
	
      return r;
    }

  %typemap(in) (int argc, VALUE* argv) {
    $1 = argc;
    $2 = argv;
  }

  VALUE values_at(int argc, VALUE* argv, ...) {
    
    VALUE r = rb_ary_new();
    ID   id = rb_intern("[]");
    swig_type_info* type = swig::type_info< Map >();
    VALUE me = SWIG_NewPointerObj( $self, type, 0 );
    for ( int i = 0; i < argc; ++i )
      {
	VALUE key = argv[i];
	VALUE tmp = rb_funcall( me, id, 1, key );
	rb_ary_push( r, tmp );
      }
    
    return r;
  }


    Map* each_key()
      {
	if ( !rb_block_given_p() )
	  rb_raise( rb_eArgError, "no block given");

	VALUE r;
	Map::iterator i = self->begin();
	Map::iterator e = self->end();
	for ( ; i != e; ++i )
	  {
	    r = swig::from( i->first );
	    rb_yield(r);
	  }
	
	return self;
      }
    
    VALUE values() {
      Map::size_type size = self->size();
      int rubysize = (size <= (Map::size_type) INT_MAX) ? (int) size : -1;
      if (rubysize < 0) {
	SWIG_RUBY_THREAD_BEGIN_BLOCK;
	rb_raise(rb_eRuntimeError, "map size not valid in Ruby");
	SWIG_RUBY_THREAD_END_BLOCK;
	return Qnil;
      }
      VALUE ary = rb_ary_new2(rubysize);
      Map::const_iterator i = self->begin();
      Map::const_iterator e = self->end();
      for ( ; i != e; ++i ) {
	rb_ary_push( ary, swig::from(i->second) );
      }
      return ary;
    }
    
    Map* each_value()
      {
	if ( !rb_block_given_p() )
	  rb_raise( rb_eArgError, "no block given");

	VALUE r;
	Map::iterator i = self->begin();
	Map::iterator e = self->end();
	for ( ; i != e; ++i )
	  {
	    r = swig::from( i->second );
	    rb_yield(r);
	  }
	
	return self;
      }

    VALUE entries() {
      Map::size_type size = self->size();
      int rubysize = (size <= (Map::size_type) INT_MAX) ? (int) size : -1;
      if (rubysize < 0) {
	SWIG_RUBY_THREAD_BEGIN_BLOCK;
	rb_raise(rb_eRuntimeError, "map size not valid in Ruby");
	SWIG_RUBY_THREAD_END_BLOCK;
	return Qnil;
      }
      VALUE ary = rb_ary_new2(rubysize);
      Map::const_iterator i = self->begin();
      Map::const_iterator e = self->end();
      for ( ; i != e; ++i ) {
	rb_ary_push( ary, swig::from<std::pair<Map::key_type, 
		     Map::mapped_type> >(*i) );
      }
      return ary;
    }
    
    bool __contains__(const key_type& key) {
      return self->find(key) != self->end();
    }

    %newobject key_iterator(VALUE *RUBY_SELF);
    swig::ConstIterator* key_iterator(VALUE *RUBY_SELF) {
      return swig::make_output_key_iterator($self->begin(), $self->begin(), 
					    $self->end(), *RUBY_SELF);
    }

    %newobject value_iterator(VALUE *RUBY_SELF);
    swig::ConstIterator* value_iterator(VALUE *RUBY_SELF) {
      return swig::make_output_value_iterator($self->begin(), $self->begin(), 
					      $self->end(), *RUBY_SELF);
    }

  }
%enddef

%define %swig_map_methods(Map...)
  %swig_map_common(Map)
  %extend {
    VALUE __getitem__(const key_type& key) const {
      Map::const_iterator i = self->find(key);
      if ( i != self->end() )
	return swig::from<Map::mapped_type>( i->second );
      else
	return Qnil;
    }

    void __setitem__(const key_type& key, const mapped_type& x) throw (std::out_of_range) {
      (*self)[key] = x;
    }

  VALUE inspect()
    {
      Map::const_iterator i = $self->begin();
      Map::const_iterator e = $self->end();
      const char *type_name = swig::type_name< Map >();
      VALUE str = rb_str_new2( type_name );
      str = rb_str_cat2( str, " {" );
      bool comma = false;
      VALUE tmp;
      for ( ; i != e; ++i, comma = true )
	{
	  if (comma) str = rb_str_cat2( str, "," );
	  tmp = swig::from< Map::key_type >( i->first );
	  tmp = rb_inspect( tmp );
	  str = rb_str_buf_append( str, tmp );
	  str = rb_str_cat2( str, "=>" );
	  tmp = swig::from< Map::mapped_type >( i->second );
	  tmp = rb_inspect( tmp );
	  str = rb_str_buf_append( str, tmp );
	}
      str = rb_str_cat2( str, "}" );
      return str;
    }

  VALUE to_a()
    {
      Map::const_iterator i = $self->begin();
      Map::const_iterator e = $self->end();
      VALUE ary = rb_ary_new2( std::distance( i, e ) );
      VALUE tmp;
      for ( ; i != e; ++i )
	{
	  // @todo: improve -- this should just be swig::from(*i)
	  tmp = swig::from< std::pair<Map::key_type, 
	    Map::mapped_type> >( *i );
	  rb_ary_push( ary, tmp );
	}
      return ary;
    }

  VALUE to_s()
    {
      Map::iterator i = $self->begin();
      Map::iterator e = $self->end();
      VALUE str = rb_str_new2( "" );
      VALUE tmp;
      for ( ; i != e; ++i )
	{
	  // @todo: improve -- this should just be swig::from(*i)
	  tmp = swig::from< std::pair<Map::key_type, 
	    Map::mapped_type> >( *i );
	  tmp = rb_obj_as_string( tmp );
	  str = rb_str_buf_append( str, tmp );
	}
      return str;
    }

  }
%enddef


%mixin std::map "Enumerable";


%rename("delete")     std::map::__delete__;
%rename("reject!")    std::map::reject_bang;
%rename("map!")       std::map::map_bang;
%rename("empty?")     std::map::empty;
%rename("include?" )  std::map::__contains__ const;
%rename("has_key?" )  std::map::has_key const;

%alias  std::map::push          "<<";


%include <std/std_map.i>
