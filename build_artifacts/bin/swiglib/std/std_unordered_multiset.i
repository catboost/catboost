//
// std::unordered_multiset
//

%include <std_unordered_set.i>

// Unordered Multiset

%define %std_unordered_multiset_methods(unordered_multiset...)
  %std_unordered_set_methods_common(unordered_multiset);
%enddef


// ------------------------------------------------------------------------
// std::unordered_multiset
// 
// const declarations are used to guess the intent of the function being
// exported; therefore, the following rationale is applied:
// 
//   -- f(std::unordered_multiset<Key>), f(const std::unordered_multiset<Key>&):
//      the parameter being read-only, either a sequence or a
//      previously wrapped std::unordered_multiset<Key> can be passed.
//   -- f(std::unordered_multiset<Key>&), f(std::unordered_multiset<Key>*):
//      the parameter may be modified; therefore, only a wrapped std::unordered_multiset
//      can be passed.
//   -- std::unordered_multiset<Key> f(), const std::unordered_multiset<Key>& f():
//      the set is returned by copy; therefore, a sequence of Key:s 
//      is returned which is most easily used in other functions
//   -- std::unordered_multiset<Key>& f(), std::unordered_multiset<Key>* f():
//      the set is returned by reference; therefore, a wrapped std::unordered_multiset
//      is returned
//   -- const std::unordered_multiset<Key>* f(), f(const std::unordered_multiset<Key>*):
//      for consistency, they expect and return a plain set pointer.
// ------------------------------------------------------------------------


// exported classes

namespace std {

  //unordered_multiset

  template <class _Key,
            class _Hash = std::hash< _Key >,
            class _Compare = std::equal_to< _Key >,
            class _Alloc = allocator< _Key > >
  class unordered_multiset {
  public:
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef _Key value_type;
    typedef _Key key_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef _Hash hasher;
    typedef _Compare key_equal;
    typedef _Alloc allocator_type;

    %traits_swigtype(_Key);

    %fragment(SWIG_Traits_frag(std::unordered_multiset< _Key, _Hash, _Compare, _Alloc >), "header",
	      fragment=SWIG_Traits_frag(_Key),
	      fragment="StdUnorderedMultisetTraits") {
      namespace swig {
	template <>  struct traits<std::unordered_multiset< _Key, _Hash, _Compare, _Alloc > > {
	  typedef pointer_category category;
	  static const char* type_name() {
	    return "std::unordered_multiset<" #_Key "," #_Hash "," #_Compare "," #_Alloc " >";
	  }
	};
      }
    }

    %typemap_traits_ptr(SWIG_TYPECHECK_MULTISET, std::unordered_multiset< _Key, _Hash, _Compare, _Alloc >);

#ifdef %swig_unordered_multiset_methods
    // Add swig/language extra methods
    %swig_unordered_multiset_methods(std::unordered_multiset< _Key, _Hash, _Compare, _Alloc >);
#endif
  
    %std_unordered_multiset_methods(unordered_multiset);
  };
}
