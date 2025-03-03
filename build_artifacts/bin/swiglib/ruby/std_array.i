/*
  std::array
*/

%fragment("StdArrayTraits","header",fragment="StdSequenceTraits")
%{
  namespace swig {
    template <class T, size_t N>
    struct traits_asptr<std::array<T, N> >  {
      static int asptr(VALUE obj, std::array<T, N> **vec) {
	return traits_asptr_stdseq<std::array<T, N> >::asptr(obj, vec);
      }
    };
    
    template <class T, size_t N>
    struct traits_from<std::array<T, N> > {
      static VALUE from(const std::array<T, N>& vec) {
	return traits_from_stdseq<std::array<T, N> >::from(vec);
      }
    };

    template <class RubySeq, class T, size_t N>
    inline void
    assign(const RubySeq& rubyseq, std::array<T, N>* seq) {
      if (rubyseq.size() < seq->size())
        throw std::invalid_argument("std::array cannot be expanded in size");
      else if (rubyseq.size() > seq->size())
        throw std::invalid_argument("std::array cannot be reduced in size");
      std::copy(rubyseq.begin(), rubyseq.end(), seq->begin());
    }

    template <class T, size_t N>
    inline void
    resize(std::array<T, N> *seq, typename std::array<T, N>::size_type n, typename std::array<T, N>::value_type x) {
      throw std::invalid_argument("std::array is a fixed size container and does not support resizing");
    }

    // Only limited slicing is supported as std::array is fixed in size
    template <class T, size_t N, class Difference>
    inline std::array<T, N>*
    getslice(const std::array<T, N>* self, Difference i, Difference j) {
      typedef std::array<T, N> Sequence;
      typename Sequence::size_type size = self->size();
      typename Sequence::size_type ii = swig::check_index(i, size, (i == size && j == size));
      typename Sequence::size_type jj = swig::slice_index(j, size);

      if (ii == 0 && jj == size) {
        Sequence *sequence = new Sequence();
        std::copy(self->begin(), self->end(), sequence->begin());
        return sequence;
      } else {
        throw std::invalid_argument("std::array object only supports getting a slice that is the size of the array");
      }
    }

    template <class T, size_t N, class Difference, class InputSeq>
    inline void
    setslice(std::array<T, N>* self, Difference i, Difference j, const InputSeq& v) {
      typedef std::array<T, N> Sequence;
      typename Sequence::size_type size = self->size();
      typename Sequence::size_type ii = swig::check_index(i, size, true);
      typename Sequence::size_type jj = swig::slice_index(j, size);

      if (ii == 0 && jj == size) {
        std::copy(v.begin(), v.end(), self->begin());
      } else {
        throw std::invalid_argument("std::array object only supports setting a slice that is the size of the array");
      }
    }

    template <class T, size_t N, class Difference>
    inline void
    delslice(std::array<T, N>* self, Difference i, Difference j) {
      throw std::invalid_argument("std::array object does not support item deletion");
    }
  }
%}


%define %swig_array_methods(Type...)
  %swig_sequence_methods_non_resizable(Type)
%enddef

%define %swig_array_methods_val(Type...)
  %swig_sequence_methods_non_resizable_val(Type);
%enddef


%mixin std::array "Enumerable";
%ignore std::array::push_back;
%ignore std::array::pop_back;


%rename("delete")     std::array::__delete__;
%rename("reject!")    std::array::reject_bang;
%rename("map!")       std::array::map_bang;
%rename("empty?")     std::array::empty;
%rename("include?" )  std::array::__contains__ const;
%rename("has_key?" )  std::array::has_key const;

%include <std/std_array.i>

