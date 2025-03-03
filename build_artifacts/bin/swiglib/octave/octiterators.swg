/* -----------------------------------------------------------------------------
 * octiterators.swg
 *
 * Users can derive form the OctSwigIterator to implement their
 * own iterators. As an example (real one since we use it for STL/STD
 * containers), the template OctSwigIterator_T does the
 * implementation for generic C++ iterators.
 * ----------------------------------------------------------------------------- */

%include <std_common.i>

%fragment("OctSwigIterator","header",fragment="<stddef.h>") {  
namespace swig {
  struct stop_iteration {
  };

  struct OctSwigIterator {
  private:
    octave_value _seq;

  protected:
    OctSwigIterator(octave_value seq) : _seq(seq)
    {
    }
      
  public:
    virtual ~OctSwigIterator() {}

    virtual octave_value value() const = 0;

    virtual OctSwigIterator *incr(size_t n = 1) = 0;

    virtual OctSwigIterator *decr(size_t n = 1)
    {
      throw stop_iteration();
    }

    virtual ptrdiff_t distance(const OctSwigIterator &x) const
    {
      throw std::invalid_argument("operation not supported");
    }

    virtual bool equal (const OctSwigIterator &x) const
    {
      throw std::invalid_argument("operation not supported");
    }
    
    virtual OctSwigIterator *copy() const = 0;

    octave_value next()
    {
      octave_value obj = value();
      incr();
      return obj;
    }

    octave_value previous()
    {
      decr();
      return value();
    }

    OctSwigIterator *advance(ptrdiff_t n)
    {
      return  (n > 0) ?  incr(n) : decr(-n);
    }
      
    bool operator == (const OctSwigIterator& x)  const
    {
      return equal(x);
    }
      
    bool operator != (const OctSwigIterator& x) const
    {
      return ! operator==(x);
    }

    OctSwigIterator* operator ++ () {
      incr();
      return this;
    }

    OctSwigIterator* operator -- () {
      decr();
      return this;
    }
      
    OctSwigIterator* operator + (ptrdiff_t n) const
    {
      return copy()->advance(n);
    }

    OctSwigIterator* operator - (ptrdiff_t n) const
    {
      return copy()->advance(-n);
    }
      
    ptrdiff_t operator - (const OctSwigIterator& x) const
    {
      return x.distance(*this);
    }
      
    static swig_type_info* descriptor() {
      static int init = 0;
      static swig_type_info* desc = 0;
      if (!init) {
	desc = SWIG_TypeQuery("swig::OctSwigIterator *");
	init = 1;
      }	
      return desc;
    }    
  };
}
}

%fragment("OctSwigIterator_T","header",fragment="<stddef.h>",fragment="OctSwigIterator",fragment="StdTraits",fragment="StdIteratorTraits") {
namespace swig {
  template<typename OutIterator>
  class OctSwigIterator_T :  public OctSwigIterator
  {
  public:
    typedef OutIterator out_iterator;
    typedef typename std::iterator_traits<out_iterator>::value_type value_type;    
    typedef OctSwigIterator_T<out_iterator> self_type;

    OctSwigIterator_T(out_iterator curr, octave_value seq)
      : OctSwigIterator(seq), current(curr)
    {
    }

    const out_iterator& get_current() const
    {
      return current;
    }

    
    bool equal (const OctSwigIterator &iter) const
    {
      const self_type *iters = dynamic_cast<const self_type *>(&iter);
      if (iters) {
	return (current == iters->get_current());
      } else {
	throw std::invalid_argument("bad iterator type");
      }
    }
    
    ptrdiff_t distance(const OctSwigIterator &iter) const
    {
      const self_type *iters = dynamic_cast<const self_type *>(&iter);
      if (iters) {
	return std::distance(current, iters->get_current());
      } else {
	throw std::invalid_argument("bad iterator type");
      }
    }    
    
  protected:
    out_iterator current;
  };
  
  template <class ValueType>
  struct from_oper 
  {
    typedef const ValueType& argument_type;
    typedef octave_value result_type;
    result_type operator()(argument_type v) const
    {
      return swig::from(v);
    }
  };

  template<typename OutIterator, 
	   typename ValueType = typename std::iterator_traits<OutIterator>::value_type,
	   typename FromOper = from_oper<ValueType> >
  class OctSwigIteratorOpen_T :  public OctSwigIterator_T<OutIterator>
  {
  public:
    FromOper from;
    typedef OutIterator out_iterator;
    typedef ValueType value_type;
    typedef OctSwigIterator_T<out_iterator>  base;
    typedef OctSwigIteratorOpen_T<OutIterator, ValueType, FromOper> self_type;
    
    OctSwigIteratorOpen_T(out_iterator curr, octave_value seq)
      : OctSwigIterator_T<OutIterator>(curr, seq)
    {
    }
    
    octave_value value() const {
      return from(static_cast<const value_type&>(*(base::current)));
    }
    
    OctSwigIterator *copy() const
    {
      return new self_type(*this);
    }

    OctSwigIterator *incr(size_t n = 1)
    {
      while (n--) {
	++base::current;
      }
      return this;
    }

    OctSwigIterator *decr(size_t n = 1)
    {
      while (n--) {
	--base::current;
      }
      return this;
    }
  };

  template<typename OutIterator, 
	   typename ValueType = typename std::iterator_traits<OutIterator>::value_type,
	   typename FromOper = from_oper<ValueType> >
  class OctSwigIteratorClosed_T :  public OctSwigIterator_T<OutIterator>
  {
  public:
    FromOper from;
    typedef OutIterator out_iterator;
    typedef ValueType value_type;
    typedef OctSwigIterator_T<out_iterator>  base;    
    typedef OctSwigIteratorClosed_T<OutIterator, ValueType, FromOper> self_type;
    
    OctSwigIteratorClosed_T(out_iterator curr, out_iterator first, out_iterator last, octave_value seq)
      : OctSwigIterator_T<OutIterator>(curr, seq), begin(first), end(last)
    {
    }
    
    octave_value value() const {
      if (base::current == end) {
	throw stop_iteration();
      } else {
	return from(static_cast<const value_type&>(*(base::current)));
      }
    }
    
    OctSwigIterator *copy() const
    {
      return new self_type(*this);
    }

    OctSwigIterator *incr(size_t n = 1)
    {
      while (n--) {
	if (base::current == end) {
	  throw stop_iteration();
	} else {
	  ++base::current;
	}
      }
      return this;
    }

    OctSwigIterator *decr(size_t n = 1)
    {
      while (n--) {
	if (base::current == begin) {
	  throw stop_iteration();
	} else {
	  --base::current;
	}
      }
      return this;
    }

  private:
    out_iterator begin;
    out_iterator end;
  };

  template<typename OutIter>
  inline OctSwigIterator*
  make_output_iterator(const OutIter& current, const OutIter& begin,const OutIter& end, octave_value seq = octave_value())
  {
    return new OctSwigIteratorClosed_T<OutIter>(current, begin, end, seq);
  }

  template<typename OutIter>
  inline OctSwigIterator*
  make_output_iterator(const OutIter& current, octave_value seq = octave_value())
  {
    return new OctSwigIteratorOpen_T<OutIter>(current, seq);
  }
}
}


%fragment("OctSwigIterator");
namespace swig 
{
// Throw a StopIteration exception
  %ignore stop_iteration;
  struct stop_iteration {};
  
  %typemap(throws) stop_iteration {
    error("stop_iteration exception");
    SWIG_fail;
  }

// Mark methods that return new objects
  %newobject OctSwigIterator::copy;
  %newobject OctSwigIterator::operator + (ptrdiff_t n) const;
  %newobject OctSwigIterator::operator - (ptrdiff_t n) const;

  %nodirector OctSwigIterator;

  %catches(swig::stop_iteration) OctSwigIterator::value() const;
  %catches(swig::stop_iteration) OctSwigIterator::incr(size_t n = 1);
  %catches(swig::stop_iteration) OctSwigIterator::decr(size_t n = 1);
  %catches(std::invalid_argument) OctSwigIterator::distance(const OctSwigIterator &x) const;
  %catches(std::invalid_argument) OctSwigIterator::equal (const OctSwigIterator &x) const;
  %catches(swig::stop_iteration) OctSwigIterator::next();
  %catches(swig::stop_iteration) OctSwigIterator::previous();
  %catches(swig::stop_iteration) OctSwigIterator::advance(ptrdiff_t n);
  %catches(swig::stop_iteration) OctSwigIterator::operator += (ptrdiff_t n);
  %catches(swig::stop_iteration) OctSwigIterator::operator -= (ptrdiff_t n);
  %catches(swig::stop_iteration) OctSwigIterator::operator + (ptrdiff_t n) const;
  %catches(swig::stop_iteration) OctSwigIterator::operator - (ptrdiff_t n) const;


  struct OctSwigIterator
  {
  protected:
    OctSwigIterator(octave_value seq);

  public:
    virtual ~OctSwigIterator();

    virtual octave_value value() const = 0;

    virtual OctSwigIterator *incr(size_t n = 1) = 0;
    
    virtual OctSwigIterator *decr(size_t n = 1);

    virtual ptrdiff_t distance(const OctSwigIterator &x) const;

    virtual bool equal (const OctSwigIterator &x) const;
    
    virtual OctSwigIterator *copy() const = 0;

    octave_value next();
    octave_value previous();
    OctSwigIterator *advance(ptrdiff_t n);

    bool operator == (const OctSwigIterator& x)  const;
    bool operator != (const OctSwigIterator& x) const;
    OctSwigIterator* operator ++ ();
    OctSwigIterator* operator -- ();
    OctSwigIterator* operator + (ptrdiff_t n) const;
    OctSwigIterator* operator - (ptrdiff_t n) const;
    ptrdiff_t operator - (const OctSwigIterator& x) const;
  };
}

