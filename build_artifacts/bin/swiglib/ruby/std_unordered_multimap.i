/*
  Multimaps
*/
%include <std_multimap.i>

%fragment("StdUnorderedMultimapTraits","header",fragment="StdMapCommonTraits")
{
  namespace swig {
    template <class RubySeq, class K, class T, class Hash, class Compare, class Alloc>
    inline void
    assign(const RubySeq& rubyseq, std::unordered_multimap<K,T,Hash,Compare,Alloc> *multimap) {
      typedef typename std::unordered_multimap<K,T,Hash,Compare,Alloc>::value_type value_type;
      typename RubySeq::const_iterator it = rubyseq.begin();
      for (;it != rubyseq.end(); ++it) {
	multimap->insert(value_type(it->first, it->second));
      }
    }

    template <class K, class T, class Hash, class Compare, class Alloc>
    struct traits_asptr<std::unordered_multimap<K,T,Hash,Compare,Alloc> >  {
      typedef std::unordered_multimap<K,T,Hash,Compare,Alloc> multimap_type;
      static int asptr(VALUE obj, std::unordered_multimap<K,T,Hash,Compare,Alloc> **val) {
	int res = SWIG_ERROR;
	if ( TYPE(obj) == T_HASH ) {
	  static ID id_to_a = rb_intern("to_a");
	  VALUE items = rb_funcall(obj, id_to_a, 0);
	  return traits_asptr_stdseq<std::unordered_multimap<K,T,Hash,Compare,Alloc>, std::pair<K, T> >::asptr(items, val);
	} else {
	  multimap_type *p;
	  res = SWIG_ConvertPtr(obj,(void**)&p,swig::type_info<multimap_type>(),0);
	  if (SWIG_IsOK(res) && val)  *val = p;
	}
	return res;
      }
    };

    template <class K, class T, class Hash, class Compare, class Alloc>
    struct traits_from<std::unordered_multimap<K,T,Hash,Compare,Alloc> >  {
      typedef std::unordered_multimap<K,T,Hash,Compare,Alloc> multimap_type;
      typedef typename multimap_type::const_iterator const_iterator;
      typedef typename multimap_type::size_type size_type;

      static VALUE from(const multimap_type& multimap) {
	swig_type_info *desc = swig::type_info<multimap_type>();
	if (desc && desc->clientdata) {
	  return SWIG_NewPointerObj(new multimap_type(multimap), desc, SWIG_POINTER_OWN);
	} else {
	  size_type size = multimap.size();
	  int rubysize = (size <= (size_type) INT_MAX) ? (int) size : -1;
	  if (rubysize < 0) {
	    SWIG_RUBY_THREAD_BEGIN_BLOCK;
	    rb_raise(rb_eRuntimeError,
		     "multimap_ size not valid in Ruby");
	    SWIG_RUBY_THREAD_END_BLOCK;
	    return Qnil;
	  }
	  VALUE obj = rb_hash_new();
	  for (const_iterator i= multimap.begin(); i!= multimap.end(); ++i) {
	    VALUE key = swig::from(i->first);
	    VALUE val = swig::from(i->second);

	    VALUE oldval = rb_hash_aref(obj, key);
	    if (oldval == Qnil) {
	      rb_hash_aset(obj, key, val);
	    } else {
	      // Multiple values for this key, create array if needed
	      // and add a new element to it.
	      VALUE ary;
	      if (TYPE(oldval) == T_ARRAY) {
		ary = oldval;
	      } else {
                ary = rb_ary_new2(2);
                rb_ary_push(ary, oldval);
                rb_hash_aset(obj, key, ary);
              }
	      rb_ary_push(ary, val);
	    }
	  }
	  return obj;
	}
      }
    };
  }
}

#define %swig_unordered_multimap_methods(MultiMap...) %swig_multimap_methods(MultiMap)

%mixin std::unordered_multimap "Enumerable";

%rename("delete")     std::unordered_multimap::__delete__;
%rename("reject!")    std::unordered_multimap::reject_bang;
%rename("map!")       std::unordered_multimap::map_bang;
%rename("empty?")     std::unordered_multimap::empty;
%rename("include?" )  std::unordered_multimap::__contains__ const;
%rename("has_key?" )  std::unordered_multimap::has_key const;

%alias  std::unordered_multimap::push          "<<";

%include <std/std_unordered_multimap.i>

