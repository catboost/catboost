/*
  Vectors
*/

%fragment("StdVectorTraits","header",fragment="StdSequenceTraits")
%{
  namespace swig {
    template <class T>
    struct traits_asptr<std::vector<T> >  {
      static int asptr(VALUE obj, std::vector<T> **vec) {
	return traits_asptr_stdseq<std::vector<T> >::asptr(obj, vec);
      }
    };
    
    template <class T>
    struct traits_from<std::vector<T> > {
      static VALUE from(const std::vector<T>& vec) {
	return traits_from_stdseq<std::vector<T> >::from(vec);
      }
    };
  }
%}



%define %swig_vector_methods(Type...) 
  %swig_sequence_methods(Type)
  %swig_sequence_front_inserters(Type);
%enddef

%define %swig_vector_methods_val(Type...) 
  %swig_sequence_methods_val(Type);
  %swig_sequence_front_inserters(Type);
%enddef


%mixin std::vector "Enumerable";
%ignore std::vector::push_back;
%ignore std::vector::pop_back;


%rename("delete")     std::vector::__delete__;
%rename("reject!")    std::vector::reject_bang;
%rename("map!")       std::vector::map_bang;
%rename("empty?")     std::vector::empty;
%rename("include?" )  std::vector::__contains__ const;
%rename("has_key?" )  std::vector::has_key const;

%alias  std::vector::push          "<<";

%include <std/std_vector.i>

