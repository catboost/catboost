/*
  Unordered Multisets
*/

%include <std_unordered_set.i>

%fragment("StdUnorderedMultisetTraits","header",fragment="StdSequenceTraits")
%{
  namespace swig {
    template <class SwigPySeq, class Key, class Hash, class Compare, class Alloc>
    inline void
    assign(const SwigPySeq& swigpyseq, std::unordered_multiset<Key,Hash,Compare,Alloc>* seq) {
      // seq->insert(swigpyseq.begin(), swigpyseq.end()); // not used as not always implemented
      typedef typename SwigPySeq::value_type value_type;
      typename SwigPySeq::const_iterator it = swigpyseq.begin();
      for (;it != swigpyseq.end(); ++it) {
	seq->insert(seq->end(),(value_type)(*it));
      }
    }

    template <class Key, class Hash, class Compare, class Alloc>
    struct traits_reserve<std::unordered_multiset<Key,Hash,Compare,Alloc> >  {
      static void reserve(std::unordered_multiset<Key,Hash,Compare,Alloc> &seq, typename std::unordered_multiset<Key,Hash,Compare,Alloc>::size_type n) {
        seq.reserve(n);
      }
    };

    template <class Key, class Hash, class Compare, class Alloc>
    struct traits_asptr<std::unordered_multiset<Key,Hash,Compare,Alloc> >  {
      static int asptr(PyObject *obj, std::unordered_multiset<Key,Hash,Compare,Alloc> **m) {
	return traits_asptr_stdseq<std::unordered_multiset<Key,Hash,Compare,Alloc> >::asptr(obj, m);
      }
    };

    template <class Key, class Hash, class Compare, class Alloc>
    struct traits_from<std::unordered_multiset<Key,Hash,Compare,Alloc> > {
      static PyObject *from(const std::unordered_multiset<Key,Hash,Compare,Alloc>& vec) {
	return traits_from_stdseq<std::unordered_multiset<Key,Hash,Compare,Alloc> >::from(vec);
      }
    };
  }
%}

#define %swig_unordered_multiset_methods(Set...) %swig_unordered_set_methods(Set)



%include <std/std_unordered_multiset.i>
