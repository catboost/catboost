/*
  Queues
*/

%fragment("StdQueueTraits","header",fragment="StdSequenceTraits")
%{
  namespace swig {
    template <class T>
    struct traits_asptr<std::queue<T> >  {
      static int asptr(VALUE obj, std::queue<T>  **vec) {
	return traits_asptr_stdseq<std::queue<T> >::asptr(obj, vec);
      }
    };

    template <class T>
    struct traits_from<std::queue<T> > {
      static VALUE from(const std::queue<T>& vec) {
	return traits_from_stdseq<std::queue<T> >::from(vec);
      }
    };
  }
%}

%rename("delete")     std::queue::__delete__;
%rename("reject!")    std::queue::reject_bang;
%rename("map!")       std::queue::map_bang;
%rename("empty?")     std::queue::empty;
%rename("include?" )  std::queue::__contains__ const;
%rename("has_key?" )  std::queue::has_key const;

%alias  std::queue::push          "<<";

%include <std/std_queue.i>
