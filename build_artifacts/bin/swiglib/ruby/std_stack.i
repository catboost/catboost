/*
  Stacks
*/

%fragment("StdStackTraits","header",fragment="StdSequenceTraits")
%{
  namespace swig {
    template <class T>
    struct traits_asptr<std::stack<T> >  {
      static int asptr(VALUE obj, std::stack<T>  **vec) {
	return traits_asptr_stdseq<std::stack<T> >::asptr(obj, vec);
      }
    };

    template <class T>
    struct traits_from<std::stack<T> > {
      static VALUE from(const std::stack<T>& vec) {
	return traits_from_stdseq<std::stack<T> >::from(vec);
      }
    };
  }
%}


%rename("delete")     std::stack::__delete__;
%rename("reject!")    std::stack::reject_bang;
%rename("map!")       std::stack::map_bang;
%rename("empty?")     std::stack::empty;
%rename("include?" )  std::stack::__contains__ const;
%rename("has_key?" )  std::stack::has_key const;

%alias  std::stack::push          "<<";


%include <std/std_stack.i>
