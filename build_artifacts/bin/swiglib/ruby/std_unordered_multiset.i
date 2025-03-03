/*
  Multisets
*/

%include <std_unordered_set.i>

%fragment("StdUnorderedMultisetTraits","header",fragment="StdUnorderedSetTraits")
%{
  namespace swig {
    template <class RubySeq, class Key, class Hash, class Compare, class Alloc>
    inline void
    assign(const RubySeq& rubyseq, std::unordered_multiset<Key,Hash,Compare,Alloc>* seq) {
      // seq->insert(rubyseq.begin(), rubyseq.end()); // not used as not always implemented
      typedef typename RubySeq::value_type value_type;
      typename RubySeq::const_iterator it = rubyseq.begin();
      for (;it != rubyseq.end(); ++it) {
	seq->insert(seq->end(),(value_type)(*it));
      }
    }

    template <class Key, class Hash, class Compare, class Alloc>
    struct traits_asptr<std::unordered_multiset<Key,Hash,Compare,Alloc> >  {
      static int asptr(VALUE obj, std::unordered_multiset<Key,Hash,Compare,Alloc> **m) {
	return traits_asptr_stdseq<std::unordered_multiset<Key,Hash,Compare,Alloc> >::asptr(obj, m);
      }
    };

    template <class Key, class Hash, class Compare, class Alloc>
    struct traits_from<std::unordered_multiset<Key,Hash,Compare,Alloc> > {
      static VALUE from(const std::unordered_multiset<Key,Hash,Compare,Alloc>& vec) {
	return traits_from_stdseq<std::unordered_multiset<Key,Hash,Compare,Alloc> >::from(vec);
      }
    };
  }
%}

#define %swig_unordered_multiset_methods(Set...) %swig_unordered_set_methods(Set)

%mixin std::unordered_multiset "Enumerable";

%rename("delete")     std::unordered_multiset::__delete__;
%rename("reject!")    std::unordered_multiset::reject_bang;
%rename("map!")       std::unordered_multiset::map_bang;
%rename("empty?")     std::unordered_multiset::empty;
%rename("include?")  std::unordered_multiset::__contains__ const;
%rename("has_key?")  std::unordered_multiset::has_key const;

%alias  std::unordered_multiset::push          "<<";

%include <std/std_unordered_multiset.i>
