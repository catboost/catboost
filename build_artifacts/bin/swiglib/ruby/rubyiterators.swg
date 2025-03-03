/* -----------------------------------------------------------------------------
 * rubyiterators.swg
 *
 * Implement a C++ 'output' iterator for Ruby.
 *
 * Users can derive form the Iterator to implement their
 * own iterators. As an example (real one since we use it for STL/STD
 * containers), the template Iterator_T does the
 * implementation for generic C++ iterators.
 * ----------------------------------------------------------------------------- */

%include <std_common.i>


%fragment("ConstIterator","header",fragment="<stddef.h>",fragment="GC_VALUE_definition") {
namespace swig {
  struct stop_iteration {
  };

  /** 
   * Abstract base class used to represent all iterators of STL containers.
   */
  struct ConstIterator {
  public:
    typedef ConstIterator self_type;

  protected:
    GC_VALUE _seq;

  protected:
    ConstIterator(VALUE seq) : _seq(seq)
    {
    }

    // Random access iterator methods, but not required in Ruby
    virtual ptrdiff_t distance(const ConstIterator &x) const
    {
      throw std::invalid_argument("distance not supported");
    }

    virtual bool equal (const ConstIterator &x) const
    {
      throw std::invalid_argument("equal not supported");
    }

    virtual self_type* advance(ptrdiff_t n)
    {
      throw std::invalid_argument("advance not supported");
    }
      
  public:
    virtual ~ConstIterator() {}

    // Access iterator method, required by Ruby
    virtual VALUE value() const {
      throw std::invalid_argument("value not supported");
      return Qnil;
    };

    virtual VALUE setValue( const VALUE& v ) {
      throw std::invalid_argument("value= not supported");
      return Qnil;
    }

    virtual self_type* next( size_t n = 1 )
    {
      return this->advance( n );
    }

    virtual self_type* previous( size_t n = 1 )
    {
      ptrdiff_t nn = n;
      return this->advance( -nn );
    }

    virtual VALUE to_s() const {
      throw std::invalid_argument("to_s not supported");
      return Qnil;
    }

    virtual VALUE inspect() const {
      throw std::invalid_argument("inspect not supported");
      return Qnil;
    }
    
    virtual ConstIterator *dup() const
    {
      throw std::invalid_argument("dup not supported");
      return NULL;
    }

    //
    // C++ common/needed methods.  We emulate a bidirectional
    // operator, to be compatible with all the STL.
    // The iterator traits will then tell the STL what type of
    // iterator we really are.
    //
    ConstIterator() : _seq( Qnil )
    {
    }

    ConstIterator( const self_type& b ) : _seq( b._seq )
    {
    }

    self_type& operator=( const self_type& b )
    {
      _seq = b._seq;
      return *this;
    }

    bool operator == (const ConstIterator& x)  const
    {
      return equal(x);
    }
      
    bool operator != (const ConstIterator& x) const
    {
      return ! operator==(x);
    }
      
    // Pre-decrement operator
    self_type& operator--()
    {
      return *previous();
    }

    // Pre-increment operator
    self_type& operator++()
    {
      return *next();
    }

    // Post-decrement operator
    self_type operator--(int)
    {
      self_type r = *this;
      previous();
      return r;
    }

    // Post-increment operator
    self_type operator++(int)
    {
      self_type r = *this;
      next();
      return r;
    }

    ConstIterator& operator += (ptrdiff_t n)
    {
      return *advance(n);
    }

    ConstIterator& operator -= (ptrdiff_t n)
    {
      return *advance(-n);
    }

    ConstIterator* operator + (ptrdiff_t n) const
    {
      return dup()->advance(n);
    }

    ConstIterator* operator - (ptrdiff_t n) const
    {
      return dup()->advance(-n);
    }
      
    ptrdiff_t operator - (const ConstIterator& x) const
    {
      return x.distance(*this);
    }
      
    static swig_type_info* descriptor() {
      static int init = 0;
      static swig_type_info* desc = 0;
      if (!init) {
	desc = SWIG_TypeQuery("swig::ConstIterator *");
	init = 1;
      }	
      return desc;
    }
  };


  /**
   * Abstract base class used to represent all non-const iterators of STL containers.
   * 
   */
  struct Iterator : public ConstIterator {
  public:
    typedef Iterator self_type;

  protected:
    Iterator(VALUE seq) : ConstIterator(seq)
    {
    }

    virtual self_type* advance(ptrdiff_t n)
    {
      throw std::invalid_argument("operation not supported");
    }

  public:
    static swig_type_info* descriptor() {
      static int init = 0;
      static swig_type_info* desc = 0;
      if (!init) {
	desc = SWIG_TypeQuery("swig::Iterator *");
	init = 1;
      }	
      return desc;
    }
    
    virtual Iterator *dup() const
    {
      throw std::invalid_argument("dup not supported");
      return NULL;
    }
      
    virtual self_type* next( size_t n = 1 )
    {
      return this->advance( n );
    }

    virtual self_type* previous( size_t n = 1 )
    {
      ptrdiff_t nn = n;
      return this->advance( -nn );
    }

    bool operator == (const ConstIterator& x)  const
    {
      return equal(x);
    }
      
    bool operator != (const Iterator& x) const
    {
      return ! operator==(x);
    }
      
    Iterator& operator += (ptrdiff_t n)
    {
      return *advance(n);
    }

    Iterator& operator -= (ptrdiff_t n)
    {
      return *advance(-n);
    }
      
    Iterator* operator + (ptrdiff_t n) const
    {
      return dup()->advance(n);
    }

    Iterator* operator - (ptrdiff_t n) const
    {
      return dup()->advance(-n);
    }
      
    ptrdiff_t operator - (const Iterator& x) const
    {
      return x.distance(*this);
    }
  };

}
}


%fragment("ConstIterator_T","header",fragment="<stddef.h>",fragment="ConstIterator",fragment="StdTraits",fragment="StdIteratorTraits") {
namespace swig {

  /** 
   * Templated base classes for all custom const_iterators.
   *
   */
  template<typename OutConstIterator>
  class ConstIterator_T :  public ConstIterator
  {
  public:
    typedef OutConstIterator const_iter;
    typedef typename std::iterator_traits<const_iter>::value_type value_type;    
    typedef ConstIterator_T<const_iter> self_type;

  protected:

    
    virtual bool equal (const ConstIterator &iter) const
    {
      const self_type *iters = dynamic_cast<const self_type *>(&iter);
      if (iters) {
	return (current == iters->get_current());
      } else {
	throw std::invalid_argument("bad iterator type");
      }
    }
    
    virtual ptrdiff_t distance(const ConstIterator &iter) const
    {
      const self_type *iters = dynamic_cast<const self_type *>(&iter);
      if (iters) {
	return std::distance(current, iters->get_current());
      } else {
	throw std::invalid_argument("bad iterator type");
      }
    }

    virtual ConstIterator* advance(ptrdiff_t n)
    {
      std::advance( current, n );
      return this;
    }

  public:
    ConstIterator_T() : ConstIterator(Qnil)
    {
    }

    ConstIterator_T(const_iter curr, VALUE seq = Qnil)
      : ConstIterator(seq), current(curr)
    {
    }

    const const_iter& get_current() const
    {
      return current;
    }

    const value_type& operator*() const
    {
      return *current;
    }

    virtual VALUE inspect() const
    {
      VALUE ret = rb_str_new2("#<");
      ret = rb_str_cat2( ret, rb_obj_classname(_seq) );
      ret = rb_str_cat2( ret, "::const_iterator " );
      VALUE cur = value();
      ret = rb_str_concat( ret, rb_inspect(cur) );
      ret = rb_str_cat2( ret, ">" );
      return ret;
    }

    virtual VALUE to_s()    const
    {
      VALUE ret = rb_str_new2( rb_obj_classname(_seq) );
      ret = rb_str_cat2( ret, "::const_iterator " );
      VALUE cur = value();
      ret = rb_str_concat( ret, rb_obj_as_string(cur) );
      return ret;
    }

  protected:
    const_iter current;
  };


  /** 
   * Templated base classes for all custom non-const iterators.
   *
   */
  template<typename InOutIterator>
  class Iterator_T :  public Iterator
  {
  public:
    typedef InOutIterator nonconst_iter;

    // Make this class iterator STL compatible, by using iterator_traits
    typedef typename std::iterator_traits<nonconst_iter >::iterator_category iterator_category;
    typedef typename std::iterator_traits<nonconst_iter >::value_type        value_type;
    typedef typename std::iterator_traits<nonconst_iter >::difference_type   difference_type;
    typedef typename std::iterator_traits<nonconst_iter >::pointer           pointer;
    typedef typename std::iterator_traits<nonconst_iter >::reference         reference;

    typedef Iterator                         base;
    typedef Iterator_T< nonconst_iter > self_type;

  protected:

    virtual bool equal (const ConstIterator &iter) const
    {
      const self_type *iters = dynamic_cast<const self_type *>(&iter);
      if (iters) {
	return (current == iters->get_current());
      } else {
	throw std::invalid_argument("bad iterator type");
      }
    }
    
    virtual ptrdiff_t distance(const ConstIterator &iter) const
    {
      const self_type *iters = dynamic_cast<const self_type *>(&iter);
      if (iters) {
	return std::distance(current, iters->get_current());
      } else {
	throw std::invalid_argument("bad iterator type");
      }
    }

    virtual Iterator* advance(ptrdiff_t n)
    {
      std::advance( current, n );
      return this;
    }

  public:

    Iterator_T(nonconst_iter curr, VALUE seq = Qnil)
      : Iterator(seq), current(curr)
    {
    }

    const nonconst_iter& get_current() const
    {
      return current;
    }

    self_type& operator=( const self_type& b )
    {
      base::operator=( b );
      return *this;
    }
    
    self_type& operator=( const value_type& b )
    {
      *current = b;
      return *this;
    }

    const value_type& operator*() const
    {
      return *current;
    }

    value_type& operator*()
    {
      return *current;
    }
    
    virtual VALUE inspect() const
    {
      VALUE ret = rb_str_new2("#<");
      ret = rb_str_cat2( ret, rb_obj_classname(_seq) );
      ret = rb_str_cat2( ret, "::iterator " );
      VALUE cur = value();
      ret = rb_str_concat( ret, rb_inspect(cur) );
      ret = rb_str_cat2( ret, ">" );
      return ret;
    }

    virtual VALUE to_s()    const
    {
      VALUE ret = rb_str_new2( rb_obj_classname(_seq) );
      ret = rb_str_cat2( ret, "::iterator " );
      VALUE cur = value();
      ret = rb_str_concat( ret, rb_obj_as_string(cur) );
      return ret;
    }

  protected:
    nonconst_iter current;
  };


  /**
   * Auxiliary functor to store the value of a ruby object inside
   * a reference of a compatible C++ type.  ie: Ruby -> C++
   * 
   */
  template <class ValueType>
  struct asval_oper 
  {
    typedef ValueType    value_type;
    typedef bool        result_type;
    bool operator()(VALUE obj, value_type& v) const
    {
      return ( swig::asval< value_type >(obj, &v) == SWIG_OK );
    }
  };

  /**
   * Auxiliary functor to return a ruby object from a C++ type. 
   * ie: C++ -> Ruby
   * 
   */
  template <class ValueType>
  struct from_oper 
  {
    typedef const ValueType& argument_type;
    typedef VALUE result_type;
    result_type operator()(argument_type v) const
    {
      return swig::from(v);
    }
  };


  /** 
   * ConstIterator class for a const_iterator with no end() boundaries.
   *
   */
  template<typename OutConstIterator, 
	   typename ValueType = typename std::iterator_traits<OutConstIterator>::value_type,
	   typename FromOper = from_oper<ValueType> >
  class ConstIteratorOpen_T :  public ConstIterator_T<OutConstIterator>
  {
  public:
    FromOper from;
    typedef OutConstIterator const_iter;
    typedef ValueType value_type;
    typedef ConstIterator_T<const_iter>  base;
    typedef ConstIteratorOpen_T<OutConstIterator, ValueType, FromOper> self_type;
    
    ConstIteratorOpen_T(const_iter curr, VALUE seq = Qnil)
      : ConstIterator_T<OutConstIterator>(curr, seq)
    {
    }
    
    virtual VALUE value() const {
      return from(static_cast<const value_type&>(*(base::current)));
    }
    
    ConstIterator *dup() const
    {
      return new self_type(*this);
    }
  };

  /** 
   * Iterator class for an iterator with no end() boundaries.
   *
   */
  template<typename InOutIterator, 
	   typename ValueType = typename std::iterator_traits<InOutIterator>::value_type,
	   typename FromOper = from_oper<ValueType>,
	   typename AsvalOper = asval_oper<ValueType> >
  class IteratorOpen_T :  public Iterator_T<InOutIterator>
  {
  public:
    FromOper  from;
    AsvalOper asval;
    typedef InOutIterator nonconst_iter;
    typedef ValueType value_type;
    typedef Iterator_T<nonconst_iter>  base;
    typedef IteratorOpen_T<InOutIterator, ValueType, FromOper, AsvalOper> self_type;

  public:
    IteratorOpen_T(nonconst_iter curr, VALUE seq = Qnil)
      : Iterator_T<InOutIterator>(curr, seq)
    {
    }
    
    virtual VALUE value() const {
      return from(static_cast<const value_type&>(*(base::current)));
    }

    virtual VALUE setValue( const VALUE& v )
    {
      value_type& dst = *base::current;
      if ( asval(v, dst) ) return v;
      return Qnil;
    }
    
    Iterator *dup() const
    {
      return new self_type(*this);
    }
  };

  /** 
   * ConstIterator class for a const_iterator where begin() and end() boundaries are known.
   *
   */
  template<typename OutConstIterator, 
	   typename ValueType = typename std::iterator_traits<OutConstIterator>::value_type,
	   typename FromOper = from_oper<ValueType> >
  class ConstIteratorClosed_T :  public ConstIterator_T<OutConstIterator>
  {
  public:
    FromOper from;
    typedef OutConstIterator const_iter;
    typedef ValueType value_type;
    typedef ConstIterator_T<const_iter>  base;    
    typedef ConstIteratorClosed_T<OutConstIterator, ValueType, FromOper> self_type;
    
  protected:
    virtual ConstIterator* advance(ptrdiff_t n)
    {
      std::advance( base::current, n );
      if ( base::current == end )
	throw stop_iteration();
      return this;
    }

  public:
    ConstIteratorClosed_T(const_iter curr, const_iter first, 
			  const_iter last, VALUE seq = Qnil)
      : ConstIterator_T<OutConstIterator>(curr, seq), begin(first), end(last)
    {
    }
    
    virtual VALUE value() const {
      if (base::current == end) {
	throw stop_iteration();
      } else {
	return from(static_cast<const value_type&>(*(base::current)));
      }
    }
    
    ConstIterator *dup() const
    {
      return new self_type(*this);
    }


  private:
    const_iter begin;
    const_iter end;
  };

  /** 
   * Iterator class for a iterator where begin() and end() boundaries are known.
   *
   */
  template<typename InOutIterator, 
	   typename ValueType = typename std::iterator_traits<InOutIterator>::value_type,
	   typename FromOper = from_oper<ValueType>,
	   typename AsvalOper = asval_oper<ValueType> >
  class IteratorClosed_T :  public Iterator_T<InOutIterator>
  {
  public:
    FromOper   from;
    AsvalOper asval;
    typedef InOutIterator nonconst_iter;
    typedef ValueType value_type;
    typedef Iterator_T<nonconst_iter>  base;
    typedef IteratorClosed_T<InOutIterator, ValueType, FromOper, AsvalOper> self_type;
    
  protected:
    virtual Iterator* advance(ptrdiff_t n)
    {
      std::advance( base::current, n );
      if ( base::current == end )
	throw stop_iteration();
      return this;
    }

  public:
    IteratorClosed_T(nonconst_iter curr, nonconst_iter first, 
		     nonconst_iter last, VALUE seq = Qnil)
      : Iterator_T<InOutIterator>(curr, seq), begin(first), end(last)
    {
    }
    
    virtual VALUE value() const {
      if (base::current == end) {
	throw stop_iteration();
      } else {
	return from(static_cast<const value_type&>(*(base::current)));
      }
    }
    
    // Iterator setter method, required by Ruby
    virtual VALUE setValue( const VALUE& v )
    {
      if (base::current == end)
	throw stop_iteration();

      value_type& dst = *base::current;
      if ( asval( v, dst ) ) return v;
      return Qnil;
    }
    
    Iterator *dup() const
    {
      return new self_type(*this);
    }

  private:
    nonconst_iter begin;
    nonconst_iter end;
  };

  /* Partial specialization for bools which don't allow de-referencing */
  template< typename InOutIterator, typename FromOper, typename AsvalOper >
  class IteratorOpen_T< InOutIterator, bool, FromOper, AsvalOper > : 
    public Iterator_T<InOutIterator>
  {
  public:
    FromOper   from;
    AsvalOper asval;
    typedef InOutIterator nonconst_iter;
    typedef bool value_type;
    typedef Iterator_T<nonconst_iter>  base;
    typedef IteratorOpen_T<InOutIterator, bool, FromOper, AsvalOper> self_type;

    IteratorOpen_T(nonconst_iter curr, VALUE seq = Qnil)
      : Iterator_T<InOutIterator>(curr, seq)
    {
    }

    virtual VALUE value() const {
      return from(static_cast<const value_type&>(*(base::current)));
    }
    
    virtual VALUE setValue( const VALUE& v )
    {
      bool tmp = *base::current;
      if ( asval( v, tmp ) )
	{
	  *base::current = tmp;
	  return v;
	}
      return Qnil;
    }    
    
    Iterator *dup() const
    {
      return new self_type(*this);
    }
    
  };

  /* Partial specialization for bools which don't allow de-referencing */
  template< typename InOutIterator, typename FromOper, typename AsvalOper >
  class IteratorClosed_T< InOutIterator, bool, FromOper, AsvalOper > : 
    public Iterator_T<InOutIterator>
  {
  public:
    FromOper   from;
    AsvalOper asval;
    typedef InOutIterator nonconst_iter;
    typedef bool value_type;
    typedef Iterator_T<nonconst_iter>  base;
    typedef IteratorClosed_T<InOutIterator, bool, FromOper, AsvalOper> self_type;
    
  protected:
    virtual Iterator* advance(ptrdiff_t n)
    {
      std::advance( base::current, n );
      if ( base::current == end )
	throw stop_iteration();
      return this;
    }

  public:
    IteratorClosed_T(nonconst_iter curr, nonconst_iter first, 
		     nonconst_iter last, VALUE seq = Qnil)
      : Iterator_T<InOutIterator>(curr, seq), begin(first), end(last)
    {
    }

    virtual VALUE value() const {
      if (base::current == end) {
	throw stop_iteration();
      } else {
	return from(static_cast<const value_type&>(*(base::current)));
      }
    }

    virtual VALUE setValue( const VALUE& v )
    {
      if (base::current == end)
	throw stop_iteration();

      bool tmp = *base::current;
      if ( asval( v, tmp ) )
	{
	  *base::current = tmp;
	  return v;
	}
      return Qnil;
    }
    
    Iterator *dup() const
    {
      return new self_type(*this);
    }

  private:
    nonconst_iter begin;
    nonconst_iter end;
  };


  /** 
   * Helper function used to wrap a bounded const_iterator.  This is to be used in
   * a %typemap(out), for example.
   *
   */
  template<typename InOutIter>
  inline Iterator*
  make_nonconst_iterator(const InOutIter& current, const InOutIter& begin,
			 const InOutIter& end, VALUE seq = Qnil)
  {
    return new IteratorClosed_T<InOutIter>(current, begin, end, seq);
  }

  /** 
   * Helper function used to wrap an unbounded const_iterator.  This is to be used in
   * a %typemap(out), for example.
   *
   */
  template<typename InOutIter>
  inline Iterator*
  make_nonconst_iterator(const InOutIter& current, VALUE seq = Qnil)
  {
    return new IteratorOpen_T<InOutIter>(current, seq);
  }

  /** 
   * Helper function used to wrap a bounded const_iterator.  This is to be used in
   * a %typemap(out), for example.
   *
   */
  template<typename OutIter>
  inline ConstIterator*
  make_const_iterator(const OutIter& current, const OutIter& begin,
                       const OutIter& end, VALUE seq = Qnil)
  {
    return new ConstIteratorClosed_T<OutIter>(current, begin, end, seq);
  }

  /** 
   * Helper function used to wrap an unbounded const_iterator.  This is to be used in
   * a %typemap(out), for example.
   *
   */
  template<typename OutIter>
  inline ConstIterator*
  make_const_iterator(const OutIter& current, VALUE seq = Qnil)
  {
    return new ConstIteratorOpen_T<OutIter>(current, seq);
  }
}
}


%fragment("ConstIterator");


//
// This part is just so SWIG is aware of the base abstract iterator class.
//
namespace swig 
{
  /*
    Throw a StopIteration exception
  */
  %ignore stop_iteration;
  struct stop_iteration {};
  
  %typemap(throws) stop_iteration {
    (void)$1;
    SWIG_Ruby_ExceptionType(NULL, Qnil);
    SWIG_fail;
  }

  /* 
     Mark methods that return new objects
  */
  %newobject ConstIterator::dup;
  %newobject ConstIterator::operator + (ptrdiff_t n) const;
  %newobject ConstIterator::operator - (ptrdiff_t n) const;

  %nodirector ConstIterator;

  %catches(swig::stop_iteration)  ConstIterator::value() const;
  %catches(swig::stop_iteration)  ConstIterator::incr(size_t n = 1);
  %catches(swig::stop_iteration)  ConstIterator::decr(size_t n = 1);
  %catches(std::invalid_argument) ConstIterator::distance(const ConstIterator &x) const;
  %catches(std::invalid_argument) ConstIterator::equal (const ConstIterator &x) const;
  %catches(swig::stop_iteration)  ConstIterator::next();
  %catches(swig::stop_iteration)  ConstIterator::previous();
  %catches(swig::stop_iteration)  ConstIterator::advance(ptrdiff_t n);
  %catches(swig::stop_iteration)  ConstIterator::operator += (ptrdiff_t n);
  %catches(swig::stop_iteration)  ConstIterator::operator -= (ptrdiff_t n);
  %catches(swig::stop_iteration)  ConstIterator::operator + (ptrdiff_t n) const;
  %catches(swig::stop_iteration)  ConstIterator::operator - (ptrdiff_t n) const;


  struct ConstIterator
  {
  protected:
    ConstIterator(VALUE seq);

  public:
    virtual ~ConstIterator();

    // Access iterator method, required by Ruby
    virtual VALUE value() const;
    
    // C++ common/needed methods
    virtual ConstIterator *dup() const;

    virtual VALUE inspect()    const;
    virtual VALUE to_s()    const;

    virtual ConstIterator* next(size_t n = 1);
    virtual ConstIterator* previous(size_t n = 1);

    bool operator == (const ConstIterator& x)  const;
    ConstIterator* operator + (ptrdiff_t n) const;
    ConstIterator* operator - (ptrdiff_t n) const;
    ptrdiff_t operator - (const ConstIterator& x) const;
  };

  struct Iterator : public ConstIterator
  {
    %rename("value=") setValue( const VALUE& v );
    virtual VALUE setValue( const VALUE& v );

    virtual Iterator *dup() const;

    virtual Iterator* next(size_t n = 1);
    virtual Iterator* previous(size_t n = 1);

    virtual VALUE inspect()    const;
    virtual VALUE to_s()    const;

    bool operator == (const Iterator& x)  const;
    Iterator* operator + (ptrdiff_t n) const;
    Iterator* operator - (ptrdiff_t n) const;
    ptrdiff_t operator - (const Iterator& x) const;
  };

}

