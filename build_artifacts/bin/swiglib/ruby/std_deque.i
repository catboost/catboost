/*
  Deques
*/

%fragment("StdDequeTraits","header",fragment="StdSequenceTraits")
%{
  namespace swig {
    template <class T>
    struct traits_asptr<std::deque<T> >  {
      static int asptr(VALUE obj, std::deque<T>  **vec) {
	return traits_asptr_stdseq<std::deque<T> >::asptr(obj, vec);
      }
    };

    template <class T>
    struct traits_from<std::deque<T> > {
      static VALUE from(const std::deque<T>& vec) {
	return traits_from_stdseq<std::deque<T> >::from(vec);
      }
    };
  }
%}

%ignore std::deque::push_back;
%ignore std::deque::pop_back;

#define %swig_deque_methods(Type...) %swig_sequence_methods(Type)
#define %swig_deque_methods_val(Type...) %swig_sequence_methods_val(Type);

%include <std/std_deque.i>
