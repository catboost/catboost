/* -----------------------------------------------------------------------------
 * octcontainer.swg
 *
 * Octave cell <-> C++ container wrapper
 *
 * This wrapper, and its iterator, allows a general use (and reuse) of
 * the mapping between C++ and Octave, thanks to the C++ templates.
 *
 * Of course, it needs the C++ compiler to support templates, but
 * since we will use this wrapper with the STL containers, that should
 * be the case.
 * ----------------------------------------------------------------------------- */

#if !defined(SWIG_NO_EXPORT_ITERATOR_METHODS)
# if !defined(SWIG_EXPORT_ITERATOR_METHODS)
#  define SWIG_EXPORT_ITERATOR_METHODS SWIG_EXPORT_ITERATOR_METHODS
# endif
#endif

%include <octiterators.swg>

// The Octave C++ Wrap

%fragment("<stdexcept>");

%include <std_except.i>

%fragment(SWIG_Traits_frag(octave_value),"header",fragment="StdTraits") {
namespace swig {
  template <>  struct traits<octave_value > {
    typedef value_category category;
    static const char* type_name() { return  "octave_value"; }
  };
  
  template <>  struct traits_from<octave_value> {
    typedef octave_value value_type;
    static octave_value from(const value_type& val) {
      return val;
    }
  };
  
  template <> 
  struct traits_check<octave_value, value_category> {
    static bool check(const octave_value&) {
      return true;
    }
  };
  
  template <>  struct traits_asval<octave_value > {   
    typedef octave_value value_type;
    static int asval(const octave_value& obj, value_type *val) {
      if (val) *val = obj;
      return SWIG_OK;
    }
  };
}
}

%fragment("OctSequence_Base","header",fragment="<stddef.h>")
{

namespace std {
  template <>
  struct less <octave_value>
  {
    bool
    operator()(const octave_value& v, const octave_value& w) const
    { 
      octave_value res = do_binary_op(octave_value::op_le,v,w);
      return res.is_true();
    }
  };
}

namespace swig {
  inline size_t
  check_index(ptrdiff_t i, size_t size, bool insert = false) {
    if ( i < 0 ) {
      if ((size_t) (-i) <= size)
	return (size_t) (i + size);
    } else if ( (size_t) i < size ) {
      return (size_t) i;
    } else if (insert && ((size_t) i == size)) {
      return size;
    }
    
    throw std::out_of_range("index out of range");
  }

  inline size_t
  slice_index(ptrdiff_t i, size_t size) {
    if ( i < 0 ) {
      if ((size_t) (-i) <= size) {
	return (size_t) (i + size);
      } else {
	throw std::out_of_range("index out of range");
      }
    } else {
      return ( (size_t) i < size ) ? ((size_t) i) : size;
    }
  }

  template <class Sequence, class Difference>
  inline typename Sequence::iterator
  getpos(Sequence* self, Difference i)  {
    typename Sequence::iterator pos = self->begin();
    std::advance(pos, check_index(i,self->size()));
    return pos;
  }

  template <class Sequence, class Difference>
  inline typename Sequence::const_iterator
  cgetpos(const Sequence* self, Difference i)  {
    typename Sequence::const_iterator pos = self->begin();
    std::advance(pos, check_index(i,self->size()));
    return pos;
  }

  template <class Sequence, class Difference>
  inline Sequence*
  getslice(const Sequence* self, Difference i, Difference j) {
    typename Sequence::size_type size = self->size();
    typename Sequence::size_type ii = swig::check_index(i, size);
    typename Sequence::size_type jj = swig::slice_index(j, size);

    if (jj > ii) {
      typename Sequence::const_iterator vb = self->begin();
      typename Sequence::const_iterator ve = self->begin();
      std::advance(vb,ii);
      std::advance(ve,jj);
      return new Sequence(vb, ve);
    } else {
      return new Sequence();
    }
  }

  template <class Sequence, class Difference, class InputSeq>
  inline void
  setslice(Sequence* self, Difference i, Difference j, const InputSeq& v) {
    typename Sequence::size_type size = self->size();
    typename Sequence::size_type ii = swig::check_index(i, size, true);
    typename Sequence::size_type jj = swig::slice_index(j, size);
    if (jj < ii) jj = ii;
    size_t ssize = jj - ii;
    if (ssize <= v.size()) {
      typename Sequence::iterator sb = self->begin();
      typename InputSeq::const_iterator vmid = v.begin();
      std::advance(sb,ii);
      std::advance(vmid, jj - ii);
      self->insert(std::copy(v.begin(), vmid, sb), vmid, v.end());
    } else {
      typename Sequence::iterator sb = self->begin();
      typename Sequence::iterator se = self->begin();
      std::advance(sb,ii);
      std::advance(se,jj);
      self->erase(sb,se);
      self->insert(sb, v.begin(), v.end());
    }
  }

  template <class Sequence, class Difference>
  inline void
  delslice(Sequence* self, Difference i, Difference j) {
    typename Sequence::size_type size = self->size();
    typename Sequence::size_type ii = swig::check_index(i, size, true);
    typename Sequence::size_type jj = swig::slice_index(j, size);
    if (jj > ii) {
      typename Sequence::iterator sb = self->begin();
      typename Sequence::iterator se = self->begin();
      std::advance(sb,ii);
      std::advance(se,jj);
      self->erase(sb,se);
    }
  }
}
}

%fragment("OctSequence_Cont","header",
	  fragment="StdTraits",
	  fragment="OctSequence_Base",
	  fragment="OctSwigIterator_T")
{
namespace swig
{
  template <class T>
    struct OctSequence_Ref // * octave can't support these, because of how assignment works
  {
    OctSequence_Ref(const octave_value& seq, int index)
      : _seq(seq), _index(index)
    {
    }
    
    operator T () const
    {
      //      swig::SwigVar_PyObject item = OctSequence_GetItem(_seq, _index);
      octave_value item; // * todo
      try {
	return swig::as<T>(item);
      } catch (const std::exception& e) {
	char msg[1024];
	sprintf(msg, "in sequence element %d ", _index);
	if (!Octave_Error_Occurred()) {
	  %type_error(swig::type_name<T>());
	}
	SWIG_Octave_AddErrorMsg(msg);
	SWIG_Octave_AddErrorMsg(e.what());
	throw;
      }
    }

    OctSequence_Ref& operator=(const T& v)
    {
      //      OctSequence_SetItem(_seq, _index, swig::from<T>(v));
      // * todo
      return *this;
    }

  private:
    octave_value _seq;
    int _index;
  };

  template <class T>
  struct OctSequence_ArrowProxy
  {
    OctSequence_ArrowProxy(const T& x): m_value(x) {}
    const T* operator->() const { return &m_value; }
    operator const T*() const { return &m_value; }
    T m_value;
  };

  template <class T, class Reference >
  struct OctSequence_InputIterator
  {
    typedef OctSequence_InputIterator<T, Reference > self;

    typedef std::random_access_iterator_tag iterator_category;
    typedef Reference reference;
    typedef T value_type;
    typedef T* pointer;
    typedef int difference_type;

    OctSequence_InputIterator()
    {
    }

    OctSequence_InputIterator(const octave_value& seq, int index)
      : _seq(seq), _index(index)
    {
    }

    reference operator*() const
    {
      return reference(_seq, _index);
    }

    OctSequence_ArrowProxy<T>
    operator->() const {
      return OctSequence_ArrowProxy<T>(operator*());
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
    octave_value _seq;
    difference_type _index;
  };

  template <class T>
  struct OctSequence_Cont
  {
    typedef OctSequence_Ref<T> reference;
    typedef const OctSequence_Ref<T> const_reference;
    typedef T value_type;
    typedef T* pointer;
    typedef int difference_type;
    typedef int size_type;
    typedef const pointer const_pointer;
    typedef OctSequence_InputIterator<T, reference> iterator;
    typedef OctSequence_InputIterator<T, const_reference> const_iterator;

    OctSequence_Cont(const octave_value& seq) : _seq(seq)
    {
      // * assert that we have map type etc.
      /*
      if (!OctSequence_Check(seq)) {
	throw std::invalid_argument("a sequence is expected");
      }
      _seq = seq;
      Py_INCREF(_seq);
      */
    }

    ~OctSequence_Cont()
    {
    }

    size_type size() const
    {
      //      return static_cast<size_type>(OctSequence_Size(_seq));
      return 0; // * todo
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

    bool check() const
    {
      int s = size();
      for (int i = 0; i < s; ++i) {
	//	swig::SwigVar_PyObject item = OctSequence_GetItem(_seq, i);
	octave_value item; // * todo
	if (!swig::check<value_type>(item))
	  return false;
      }
      return true;
    }

  private:
    octave_value _seq;
  };

}
}

%define %swig_sequence_iterator(Sequence...)
#if defined(SWIG_EXPORT_ITERATOR_METHODS)
  class iterator;
  class reverse_iterator;
  class const_iterator;
  class const_reverse_iterator;

  %typemap(out,noblock=1,fragment="OctSequence_Cont")
    iterator, reverse_iterator, const_iterator, const_reverse_iterator {
    $result = SWIG_NewPointerObj(swig::make_output_iterator(%static_cast($1,const $type &)),
				 swig::OctSwigIterator::descriptor(),SWIG_POINTER_OWN);
  }
  %typemap(out,fragment="OctSequence_Cont")
    std::pair<iterator, iterator>, std::pair<const_iterator, const_iterator> {
    octave_value_list tmpc;
    tmpc.append(SWIG_NewPointerObj(swig::make_output_iterator(%static_cast($1,const $type &).first),
				   swig::OctSwigIterator::descriptor(),SWIG_POINTER_OWN));
    tmpc.append(SWIG_NewPointerObj(swig::make_output_iterator(%static_cast($1,const $type &).second),
				   swig::OctSwigIterator::descriptor(),SWIG_POINTER_OWN));
    $result = Cell(tmpc);
  }

  %fragment("SwigPyPairBoolOutputIterator","header",fragment=SWIG_From_frag(bool),fragment="OctSequence_Cont") {}

  %typemap(out,fragment="OctPairBoolOutputIterator")
    std::pair<iterator, bool>, std::pair<const_iterator, bool> {
    octave_value_list tmpc;
    tmpc.append(SWIG_NewPointerObj(swig::make_output_iterator(%static_cast($1,const $type &).first),
				   swig::OctSwigIterator::descriptor(),SWIG_POINTER_OWN));
    tmpc.append(SWIG_From(bool)(%static_cast($1,const $type &).second));
    $result = Cell(tmpc);
  }

  %typemap(in,noblock=1,fragment="OctSequence_Cont")
    iterator(swig::OctSwigIterator *iter = 0, int res),
    reverse_iterator(swig::OctSwigIterator *iter = 0, int res),
    const_iterator(swig::OctSwigIterator *iter = 0, int res),
    const_reverse_iterator(swig::OctSwigIterator *iter = 0, int res) {
    res = SWIG_ConvertPtr($input, %as_voidptrptr(&iter), swig::OctSwigIterator::descriptor(), 0);
    if (!SWIG_IsOK(res) || !iter) {
      %argument_fail(SWIG_TypeError, "$type", $symname, $argnum);
    } else {
      swig::OctSwigIterator_T<$type > *iter_t = dynamic_cast<swig::OctSwigIterator_T<$type > *>(iter);
      if (iter_t) {
	$1 = iter_t->get_current();
      } else {
	%argument_fail(SWIG_TypeError, "$type", $symname, $argnum);
      }
    }
  }

  %typecheck(%checkcode(ITERATOR),noblock=1,fragment="OctSequence_Cont")
    iterator, reverse_iterator, const_iterator, const_reverse_iterator {
    swig::OctSwigIterator *iter = 0;
    int res = SWIG_ConvertPtr($input, %as_voidptrptr(&iter), swig::OctSwigIterator::descriptor(), 0);
    $1 = (SWIG_IsOK(res) && iter && (dynamic_cast<swig::OctSwigIterator_T<$type > *>(iter) != 0));
  }

  %fragment("OctSequence_Cont");
#endif //SWIG_EXPORT_ITERATOR_METHODS
%enddef

// The octave container methods

%define %swig_container_methods(Container...)
%enddef

%define %swig_sequence_methods_common(Sequence...)
  %swig_sequence_iterator(%arg(Sequence))
  %swig_container_methods(%arg(Sequence))
  
  %fragment("OctSequence_Base");

  %extend {
    value_type pop() throw (std::out_of_range) {
      if (self->size() == 0)
	throw std::out_of_range("pop from empty container");
      Sequence::value_type x = self->back();
      self->pop_back();
      return x;
    }

    value_type __paren__(difference_type i) throw (std::out_of_range) {
      return *(swig::cgetpos(self, i));
    }

    void __paren_asgn__(difference_type i, value_type x) throw (std::out_of_range) {
      *(swig::getpos(self,i)) = x;
    }

    void append(value_type x) {
      self->push_back(x);
    }
  }

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
	  fragment="OctSequence_Cont")
{
namespace swig {
  template <class OctSeq, class Seq>
  inline void
  assign(const OctSeq& octseq, Seq* seq) {
%#ifdef SWIG_STD_NOASSIGN_STL
    typedef typename OctSeq::value_type value_type;
    typename OctSeq::const_iterator it = octseq.begin();
    for (;it != octseq.end(); ++it) {
      seq->insert(seq->end(),(value_type)(*it));
    }
%#else
    seq->assign(octseq.begin(), octseq.end());
%#endif
  }

  template <class Seq, class T = typename Seq::value_type >
  struct traits_asptr_stdseq {
    typedef Seq sequence;
    typedef T value_type;

    static int asptr(const octave_value& obj, sequence **seq) {
      if (!obj.is_defined() || Swig::swig_value_deref(obj)) {
	sequence *p;
	swig_type_info *descriptor = swig::type_info<sequence>();
	if (descriptor && SWIG_IsOK(SWIG_ConvertPtr(obj, (void **)&p, descriptor, 0))) {
	  if (seq) *seq = p;
	  return SWIG_OLDOBJ;
	}
%#if SWIG_OCTAVE_PREREQ(4,4,0)
      } else if (obj.iscell()) {
%#else
      } else if (obj.is_cell()) {
%#endif
	try {
	  OctSequence_Cont<value_type> octseq(obj);
	  if (seq) {
	    sequence *pseq = new sequence();
	    assign(octseq, pseq);
	    *seq = pseq;
	    return SWIG_NEWOBJ;
	  } else {
	    return octseq.check() ? SWIG_OK : SWIG_ERROR;
	  }
	} catch (std::exception& e) {
	  if (seq&&!error_state)
	    error("swig type error: %s",e.what());
	  return SWIG_ERROR;
	}
      }
      return SWIG_ERROR;
    }
  };

  template <class Seq, class T = typename Seq::value_type >
  struct traits_from_stdseq {
    typedef Seq sequence;
    typedef T value_type;
    typedef typename Seq::size_type size_type;
    typedef typename sequence::const_iterator const_iterator;

    static octave_value from(const sequence& seq) {
#ifdef SWIG_OCTAVE_EXTRA_NATIVE_CONTAINERS
      swig_type_info *desc = swig::type_info<sequence>();
      if (desc && desc->clientdata) {
	return SWIG_NewPointerObj(new sequence(seq), desc, SWIG_POINTER_OWN);
      }
#endif
      size_type size = seq.size();
      if (size <= (size_type)INT_MAX) {
	Cell c(size,1);
	int i = 0;
	for (const_iterator it = seq.begin();
	     it != seq.end(); ++it, ++i) {
	  c(i) = swig::from<value_type>(*it);
	}
	return c;
      } else {
	error("swig overflow error: sequence size not valid in octave");
	return octave_value();
      }
      return octave_value();
    }
  };
}
}

