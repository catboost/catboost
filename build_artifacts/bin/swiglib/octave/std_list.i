// Lists

%fragment("StdListTraits","header",fragment="StdSequenceTraits")
%{
  namespace swig {
    template <class T >
    struct traits_asptr<std::list<T> >  {
      static int asptr(const octave_value& obj, std::list<T> **lis) {
	return traits_asptr_stdseq<std::list<T> >::asptr(obj, lis);
      }
    };

    template <class T>
    struct traits_from<std::list<T> > {
      static octave_value *from(const std::list<T>& vec) {
	return traits_from_stdseq<std::list<T> >::from(vec);
      }
    };
  }
%}

#define %swig_list_methods(Type...) %swig_sequence_methods(Type)
#define %swig_list_methods_val(Type...) %swig_sequence_methods_val(Type);

%include <std/std_list.i>

