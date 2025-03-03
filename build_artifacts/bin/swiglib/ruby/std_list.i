/*
  Lists
*/

%fragment("StdListTraits","header",fragment="StdSequenceTraits")
%{
  namespace swig {
    template <class T >
    struct traits_asptr<std::list<T> >  {
      static int asptr(VALUE obj, std::list<T> **lis) {
	return traits_asptr_stdseq<std::list<T> >::asptr(obj, lis);
      }
    };

    template <class T>
    struct traits_from<std::list<T> > {
      static VALUE from(const std::list<T>& vec) {
	return traits_from_stdseq<std::list<T> >::from(vec);
      }
    };
  }
%}

%ignore std::list::push_back;
%ignore std::list::pop_back;

#define %swig_list_methods(Type...) %swig_sequence_methods(Type)
#define %swig_list_methods_val(Type...) %swig_sequence_methods_val(Type);

%mixin std::list "Enumerable";

%rename("delete")     std::list::__delete__;
%rename("reject!")    std::list::reject_bang;
%rename("map!")       std::list::map_bang;
%rename("empty?")     std::list::empty;
%rename("include?" )  std::list::__contains__ const;
%rename("has_key?" )  std::list::has_key const;

%alias  std::list::push          "<<";

%include <std/std_list.i>

