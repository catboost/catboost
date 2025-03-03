/*
  Multisets
*/

%include <std_set.i>

%fragment("StdMultisetTraits","header",fragment="StdSequenceTraits")
%{
  namespace swig {
    template <class RubySeq, class T> 
    inline void
    assign(const RubySeq& rubyseq, std::multiset<T>* seq) {
      // seq->insert(rubyseq.begin(), rubyseq.end()); // not used as not always implemented
      typedef typename RubySeq::value_type value_type;
      typename RubySeq::const_iterator it = rubyseq.begin();
      for (;it != rubyseq.end(); ++it) {
	seq->insert(seq->end(),(value_type)(*it));
      }
    }

    template <class T>
    struct traits_asptr<std::multiset<T> >  {
      static int asptr(VALUE obj, std::multiset<T> **m) {
	return traits_asptr_stdseq<std::multiset<T> >::asptr(obj, m);
      }
    };

    template <class T>
    struct traits_from<std::multiset<T> > {
      static VALUE from(const std::multiset<T>& vec) {
	return traits_from_stdseq<std::multiset<T> >::from(vec);
      }
    };
  }
%}

#define %swig_multiset_methods(Set...) %swig_set_methods(Set)

%mixin std::multiset "Enumerable";

%rename("delete")     std::multiset::__delete__;
%rename("reject!")    std::multiset::reject_bang;
%rename("map!")       std::multiset::map_bang;
%rename("empty?")     std::multiset::empty;
%rename("include?" )  std::multiset::__contains__ const;
%rename("has_key?" )  std::multiset::has_key const;

%alias  std::multiset::push          "<<";

%include <std/std_multiset.i>
