
//
// Common fragments
//


/**** The python container methods  ****/



%fragment("StdSequenceTraits","header",fragment="<stddef.h>")
{
%#include <functional>
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

%define %swig_container_methods(Container...)

  %newobject __getslice__;

  %extend {
    bool __nonzero__() const {
      return !(self->empty());
    }

    size_type __len__() const {
      return self->size();
    }
  }
%enddef

%define %swig_sequence_methods_common(Sequence...)
//  %swig_sequence_iterator(%arg(Sequence))
  %swig_container_methods(%arg(Sequence))

  %fragment("StdSequenceTraits");

  %extend {
    value_type pop() throw (std::out_of_range) {
      if (self->size() == 0)
	throw std::out_of_range("pop from empty container");
      Sequence::value_type x = self->back();
      self->pop_back();
      return x;
    }

    Sequence* __getslice__(difference_type i, difference_type j) throw (std::out_of_range) {
      return swig::getslice(self, i, j);
    }

    void __setslice__(difference_type i, difference_type j, const Sequence& v) 
      throw (std::out_of_range, std::invalid_argument) {
      swig::setslice(self, i, j, v);
    }

    void __delslice__(difference_type i, difference_type j) throw (std::out_of_range) {
      swig::delslice(self, i, j);
    }

    void __delitem__(difference_type i) throw (std::out_of_range) {
      self->erase(swig::getpos(self,i));
    }
  }
%enddef

%define %swig_sequence_methods(Sequence...)
  %swig_sequence_methods_common(%arg(Sequence))
  %extend {
    const value_type& __getitem__(difference_type i) const throw (std::out_of_range) {
      return *(swig::cgetpos(self, i));
    }

    void __setitem__(difference_type i, const value_type& x) throw (std::out_of_range) {
      *(swig::getpos(self,i)) = x;
    }

    void append(const value_type& x) {
      self->push_back(x);
    }
 }
%enddef

%define %swig_sequence_methods_val(Sequence...)
  %swig_sequence_methods_common(%arg(Sequence))
  %extend {
    value_type __getitem__(difference_type i) throw (std::out_of_range) {
      return *(swig::cgetpos(self, i));
    }

    void __setitem__(difference_type i, value_type x) throw (std::out_of_range) {
      *(swig::getpos(self,i)) = x;
    }

    void append(value_type x) {
      self->push_back(x);
    }
 }
%enddef
