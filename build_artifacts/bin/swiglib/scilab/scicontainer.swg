/* -----------------------------------------------------------------------------
 * scicontainer.swg
 *
 * Scilab list <-> C++ container wrapper
 *
 * This wrapper, and its iterator, allows a general use (and reuse) of
 * the mapping between C++ and Scilab, thanks to the C++ templates.
 *
 * Of course, it needs the C++ compiler to support templates, but
 * since we will use this wrapper with the STL containers, that should
 * be the case.
 * ----------------------------------------------------------------------------- */

%{
#include <iostream>
%}

#if !defined(SWIG_NO_EXPORT_ITERATOR_METHODS)
# if !defined(SWIG_EXPORT_ITERATOR_METHODS)
#  define SWIG_EXPORT_ITERATOR_METHODS SWIG_EXPORT_ITERATOR_METHODS
# endif
#endif


// #define (SWIG_SCILAB_EXTRA_NATIVE_CONTAINERS)
// if defined: sequences in return are converted from/to Scilab lists or matrices
// if not defined: sequences are passed from/to Scilab as pointers

%{
#define SWIG_STD_NOASSIGN_STL
%}

%include <sciiterators.swg>
%include <scisequence.swg>

%{
#include <stdexcept>
%}

%include <exception.i>
%include <std_except.i>

%fragment("SciSequence_Cont", "header",
	  fragment="StdTraits",
	  fragment="SwigSciIterator_T",
	  fragment=SWIG_Traits_Sequence_frag(ptr),
	  fragment=SWIG_Traits_SequenceItem_frag(ptr))
{
namespace swig
{
  template <class T>
  struct SciSequence_Ref
  {
    SciSequence_Ref(const SwigSciObject& seq, int index)
      : _seq(seq), _index(index)
    {
      if (traits_as_sequence<T>::get(_seq, &piSeqAddr) != SWIG_OK)
      {
        throw std::invalid_argument("Cannot get sequence data.");
      }
    }

    operator T () const
    {
      return traits_asval_sequenceitem<T>::asval(_seq, piSeqAddr, _index);
    }

    SciSequence_Ref& operator=(const T& v)
    {
      // TODO
      return *this;
    }

    private:
      SwigSciObject _seq;
      int _index;
      void *piSeqAddr;
  };


  template <class T>
  struct SciSequence_ArrowProxy
  {
    SciSequence_ArrowProxy(const T& x): m_value(x) {}
    const T* operator->() const { return &m_value; }
    operator const T*() const { return &m_value; }
    T m_value;
  };

  template <class T, class Reference >
  struct SwigSciSequence_InputIterator
  {
    typedef SwigSciSequence_InputIterator<T, Reference > self;

    typedef std::random_access_iterator_tag iterator_category;
    typedef Reference reference;
    typedef T value_type;
    typedef T* pointer;
    typedef int difference_type;

    SwigSciSequence_InputIterator()
    {
    }

    SwigSciSequence_InputIterator(const SwigSciObject& seq, int index)
      : _seq(seq), _index(index)
    {
    }

    reference operator*() const
    {
      return reference(_seq, _index);
    }

    SciSequence_ArrowProxy<T>
    operator->() const {
      return SciSequence_ArrowProxy<T>(operator*());
    }

    bool operator==(const self& ri) const
    {
      return (_index == ri._index);
    }

    bool operator!=(const self& ri) const
    {
      return !(operator==(ri));
    }

    self& operator ++ ()
    {
      ++_index;
      return *this;
    }

    self& operator -- ()
    {
      --_index;
      return *this;
    }

    self& operator += (difference_type n)
    {
      _index += n;
      return *this;
    }

    self operator +(difference_type n) const
    {
      return self(_seq, _index + n);
    }

    self& operator -= (difference_type n)
    {
      _index -= n;
      return *this;
    }

    self operator -(difference_type n) const
    {
      return self(_seq, _index - n);
    }

    difference_type operator - (const self& ri) const
    {
      return _index - ri._index;
    }

    bool operator < (const self& ri) const
    {
      return _index < ri._index;
    }

    reference
    operator[](difference_type n) const
    {
      return reference(_seq, _index + n);
    }

  private:
    SwigSciObject _seq;
    difference_type _index;
  };

  template <class T>
  struct SciSequence_Cont
  {
    typedef SciSequence_Ref<T> reference;
    typedef const SciSequence_Ref<T> const_reference;
    typedef T value_type;
    typedef T* pointer;
    typedef int difference_type;
    typedef int size_type;
    typedef const pointer const_pointer;
    typedef SwigSciSequence_InputIterator<T, reference> iterator;
    typedef SwigSciSequence_InputIterator<T, const_reference> const_iterator;

    SciSequence_Cont(const SwigSciObject& seq) : _seq(seq)
    {
    }

    ~SciSequence_Cont()
    {
    }

    size_type size() const
    {
      int iSeqSize;
      if (traits_as_sequence<value_type>::size(_seq, &iSeqSize) == SWIG_OK)
      {
        return iSeqSize;
      }
      else
      {
        return SWIG_ERROR;
      }
    }

    bool empty() const
    {
      return size() == 0;
    }

    iterator begin()
    {
      return iterator(_seq, 0);
    }

    const_iterator begin() const
    {
      return const_iterator(_seq, 0);
    }

    iterator end()
    {
      return iterator(_seq, size());
    }

    const_iterator end() const
    {
      return const_iterator(_seq, size());
    }

    reference operator[](difference_type n)
    {
      return reference(_seq, n);
    }

    const_reference operator[](difference_type n)  const
    {
      return const_reference(_seq, n);
    }

  private:
    SwigSciObject _seq;
  };
}
}

%define %swig_sequence_iterator(Sequence...)
#if defined(SWIG_EXPORT_ITERATOR_METHODS)
  class iterator;
  class reverse_iterator;
  class const_iterator;
  class const_reverse_iterator;

  %typemap(out,noblock=1,fragment="SciSequence_Cont")
    iterator, reverse_iterator, const_iterator, const_reverse_iterator {
    %set_output(SWIG_NewPointerObj(swig::make_output_iterator(%static_cast($1,const $type &)),
           swig::SciSwigIterator::descriptor(),SWIG_POINTER_OWN));
  }
  %typemap(out,fragment="SciSequence_Cont")
    std::pair<iterator, iterator>, std::pair<const_iterator, const_iterator> {
    // TODO: return a Scilab list from the pair (see code for Octave)
  }

  %fragment("SciSwigPairBoolOutputIterator", "header",
    fragment=SWIG_From_frag(bool), fragment="SciSequence_Cont") {}

  %typemap(out,fragment="SciSwigPairBoolOutputIterator")
    std::pair<iterator, bool>, std::pair<const_iterator, bool> {
    // TODO: return a Scilab list from the pair (see code for Octave)
  }

  %typemap(in,noblock=1,fragment="SciSequence_Cont")
    iterator(swig::SciSwigIterator *iter = 0, int res),
    reverse_iterator(swig::SciSwigIterator *iter = 0, int res),
    const_iterator(swig::SciSwigIterator *iter = 0, int res),
    const_reverse_iterator(swig::SciSwigIterator *iter = 0, int res) {
    res = SWIG_ConvertPtr((SwigSciObject)$input, %as_voidptrptr(&iter), swig::SciSwigIterator::descriptor(), 0);
    if (!SWIG_IsOK(res) || !iter) {
      %argument_fail(SWIG_TypeError, "$type", $symname, $argnum);
    } else {
      swig::SwigSciIterator_T<$type > *iter_t = dynamic_cast<swig::SwigSciIterator_T<$type > *>(iter);
      if (iter_t) {
	$1 = iter_t->get_current();
      } else {
	%argument_fail(SWIG_TypeError, "$type", $symname, $argnum);
      }
    }
  }

  %typecheck(%checkcode(ITERATOR),noblock=1,fragment="SciSequence_Cont")
    iterator, reverse_iterator, const_iterator, const_reverse_iterator {
    swig::SciSwigIterator *iter = 0;
    int res = SWIG_ConvertPtr((SwigSciObject)$input, %as_voidptrptr(&iter), swig::SciSwigIterator::descriptor(), 0);
    $1 = (SWIG_IsOK(res) && iter && (dynamic_cast<swig::SwigSciIterator_T<$type > *>(iter) != 0));
  }

  %fragment("SciSequence_Cont");
#endif //SWIG_EXPORT_ITERATOR_METHODS
%enddef

// The Scilab container methods

%define %swig_container_methods(Container...)
%enddef

%define %swig_sequence_methods_common(Sequence...)
  %swig_sequence_iterator(%arg(Sequence))
  %swig_container_methods(%arg(Sequence))

%enddef

%define %swig_sequence_methods(Sequence...)
  %swig_sequence_methods_common(%arg(Sequence))
%enddef

%define %swig_sequence_methods_val(Sequence...)
  %swig_sequence_methods_common(%arg(Sequence))
%enddef

//
// Common fragments
//

%fragment("StdSequenceTraits","header",
	  fragment="StdTraits",
	  fragment="SciSequence_Cont")
{
namespace swig {
  template <class SciSeq, class Seq>
  inline void
  assign(const SciSeq& sciSeq, Seq* seq) {
%#ifdef SWIG_STD_NOASSIGN_STL
    typedef typename SciSeq::value_type value_type;
    typename SciSeq::const_iterator it = sciSeq.begin();
    for (;it != sciSeq.end(); ++it) {
      seq->insert(seq->end(),(value_type)(*it));
    }
%#else
    seq->assign(sciSeq.begin(), sciSeq.end());
%#endif
  }

  template <class Seq, class T = typename Seq::value_type >
  struct traits_asptr_stdseq {
    typedef Seq sequence;
    typedef T value_type;

    static int asptr(const SwigSciObject& obj, sequence **seq)
    {
      swig_type_info *typeInfo = swig::type_info<sequence>();
      if (typeInfo)
      {
          sequence *p;
          if (SWIG_ConvertPtr(obj, (void**)&p, typeInfo, 0) == SWIG_OK)
          {
            if (seq)
              *seq = p;
            return SWIG_OLDOBJ;
          }
      }

      if (traits_as_sequence<value_type>::check(obj) == SWIG_OK)
      {
        try
        {
          SciSequence_Cont<value_type> sciSeq(obj);
          if (seq)
          {
            *seq = new sequence();
            assign(sciSeq, *seq);
            return SWIG_NEWOBJ;
          }
          else
          {
            return SWIG_ERROR;
          }
        }
        catch (std::exception& e)
        {
          SWIG_exception(SWIG_RuntimeError, e.what());
          return SWIG_ERROR;
        }
      }
      else 
      {
        return SWIG_ERROR;
      }
    }
  };

  template <class Seq, class T = typename Seq::value_type >
  struct traits_from_stdseq {
    typedef Seq sequence;
    typedef T value_type;
    typedef typename Seq::size_type size_type;
    typedef typename sequence::const_iterator const_iterator;

    static SwigSciObject from(const sequence& seq)
    {
      %#ifdef SWIG_SCILAB_EXTRA_NATIVE_CONTAINERS
      swig_type_info *typeInfo = swig::type_info<sequence>();
      if (typeInfo)
      {
        return SWIG_NewPointerObj(new sequence(seq), typeInfo, SWIG_POINTER_OWN);
      }
      %#endif

      try
      {
        void *data;
        size_type size = seq.size();
        if (traits_from_sequence<value_type>::create(size, &data) == SWIG_OK) {
          const_iterator it;
          int index = 0;
          for (it = seq.begin(); it != seq.end(); ++it)
          {
            traits_from_sequenceitem<value_type>::from(data, index, *it);
            index++;
          }
          return traits_from_sequence<value_type>::set(size, data);
        }
        return SWIG_OK;
      }
      catch (std::exception& e)
      {
        SWIG_exception(SWIG_RuntimeError, e.what());
        return SWIG_ERROR;
      }
    }
  };
}
}
