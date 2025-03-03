/*
  Vectors + allocators
*/

%fragment("StdVectorATraits","header",fragment="StdSequenceTraits")
%{
  namespace swig {
    template <class T, class A>
      struct traits_asptr<std::vector<T,A> >  {
      typedef std::vector<T,A> vector_type;
      typedef T value_type;
      static int asptr(VALUE obj, vector_type **vec) {
	return traits_asptr_stdseq<vector_type>::asptr(obj, vec);
      }
    };

    template <class T, class A>
    struct traits_from<std::vector<T,A> > {
      typedef std::vector<T,A> vector_type;
      static VALUE from(const vector_type& vec) {
	return traits_from_stdseq<vector_type>::from(vec);
      }
    };
  }
%}


#define %swig_vector_methods(Type...) %swig_sequence_methods(Type)
#define %swig_vector_methods_val(Type...) %swig_sequence_methods_val(Type);

%mixin std::vector "Enumerable";
%ignore std::vector::push_back;
%ignore std::vector::pop_back;
%alias  std::vector::push "<<";

%include <std/std_vectora.i>
