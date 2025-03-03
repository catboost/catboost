/*
 *
 * C++ type : STL deque
 * Scilab type : matrix (for primitive types) or list (for pointer types)
 *
*/

%fragment("StdDequeTraits", "header", fragment="StdSequenceTraits")
%{
  namespace swig {
    template <class T>
    struct traits_asptr<std::deque<T> >  {
      static int asptr(const SwigSciObject &obj, std::deque<T> **deq) {
        return traits_asptr_stdseq<std::deque<T> >::asptr(obj, deq);
      }
    };

    template <class T>
    struct traits_from<std::deque<T> > {
      static SwigSciObject from(const std::deque<T>& deq) {
	      return traits_from_stdseq<std::deque<T> >::from(deq);
      }
    };
  }
%}


#define %swig_deque_methods(Type...) %swig_sequence_methods(Type)
#define %swig_deque_methods_val(Type...) %swig_sequence_methods_val(Type);

%include <std/std_deque.i>
