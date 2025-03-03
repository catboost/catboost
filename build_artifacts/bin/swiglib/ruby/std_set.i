/*
  Sets
*/

%fragment("StdSetTraits","header",fragment="<stddef.h>",fragment="StdSequenceTraits")
%{
  namespace swig {
    template <class RubySeq, class T> 
    inline void 
    assign(const RubySeq& rubyseq, std::set<T>* seq) {
      // seq->insert(rubyseq.begin(), rubyseq.end()); // not used as not always implemented
      typedef typename RubySeq::value_type value_type;
      typename RubySeq::const_iterator it = rubyseq.begin();
      for (;it != rubyseq.end(); ++it) {
	seq->insert(seq->end(),(value_type)(*it));
      }
    }

    template <class T>
    struct traits_asptr<std::set<T> >  {
      static int asptr(VALUE obj, std::set<T> **s) {  
	return traits_asptr_stdseq<std::set<T> >::asptr(obj, s);
      }
    };

    template <class T>
    struct traits_from<std::set<T> > {
      static VALUE from(const std::set<T>& vec) {
	return traits_from_stdseq<std::set<T> >::from(vec);
      }
    };


    /** 
     * Set Iterator class for an iterator with no end() boundaries.
     *
     */
    template<typename InOutIterator, 
	     typename ValueType = typename std::iterator_traits<InOutIterator>::value_type,
	     typename FromOper = from_oper<ValueType>,
	     typename AsvalOper = asval_oper<ValueType> >
      class SetIteratorOpen_T :  public Iterator_T<InOutIterator>
    {
    public:
      FromOper  from;
      AsvalOper asval;
      typedef InOutIterator nonconst_iter;
      typedef ValueType value_type;
      typedef Iterator_T<nonconst_iter>  base;
      typedef SetIteratorOpen_T<InOutIterator, ValueType, FromOper, AsvalOper> self_type;

    public:
      SetIteratorOpen_T(nonconst_iter curr, VALUE seq = Qnil)
	: Iterator_T<InOutIterator>(curr, seq)
      {
      }
    
      virtual VALUE value() const {
	return from(static_cast<const value_type&>(*(base::current)));
      }

      // no setValue allowed
    
      Iterator *dup() const
      {
	return new self_type(*this);
      }
    };


    /** 
     * Set Iterator class for a iterator where begin() and end() boundaries
       are known.
     *
     */
    template<typename InOutIterator, 
	     typename ValueType = typename std::iterator_traits<InOutIterator>::value_type,
	     typename FromOper = from_oper<ValueType>,
	     typename AsvalOper = asval_oper<ValueType> >
    class SetIteratorClosed_T :  public Iterator_T<InOutIterator>
    {
    public:
      FromOper   from;
      AsvalOper asval;
      typedef InOutIterator nonconst_iter;
      typedef ValueType value_type;
      typedef Iterator_T<nonconst_iter>  base;
      typedef SetIteratorClosed_T<InOutIterator, ValueType, FromOper, AsvalOper> self_type;
    
    protected:
      virtual Iterator* advance(ptrdiff_t n)
      {
	std::advance( base::current, n );
	if ( base::current == end )
	  throw stop_iteration();
	return this;
      }

    public:
      SetIteratorClosed_T(nonconst_iter curr, nonconst_iter first, 
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

      // no setValue allowed
    
    
      Iterator *dup() const
      {
	return new self_type(*this);
      }

    private:
      nonconst_iter begin;
      nonconst_iter end;
    };

    // Template specialization to construct a closed iterator for sets
    // this turns a nonconst iterator into a const one for ruby to avoid
    // allowing the user to change the value
    template< typename InOutIter >
    inline Iterator*
    make_set_nonconst_iterator(const InOutIter& current, 
			       const InOutIter& begin,
			       const InOutIter& end, 
			       VALUE seq = Qnil)
    {
      return new SetIteratorClosed_T< InOutIter >(current, 
						  begin, end, seq);
    }

    // Template specialization to construct an open iterator for sets
    // this turns a nonconst iterator into a const one for ruby to avoid
    // allowing the user to change the value
    template< typename InOutIter >
    inline Iterator*
    make_set_nonconst_iterator(const InOutIter& current, 
			       VALUE seq = Qnil)
    {
      return new SetIteratorOpen_T< InOutIter >(current, seq);
    }

  }
%}

%define %swig_sequence_methods_extra_set(Sequence...)
  %extend {
    %alias reject_bang "delete_if";
    Sequence* reject_bang() {
      if ( !rb_block_given_p() )
	rb_raise( rb_eArgError, "no block given" );

      for ( Sequence::iterator i = $self->begin(); i != $self->end(); ) {
        VALUE r = swig::from< Sequence::value_type >(*i);
        Sequence::iterator current = i++;
        if ( RTEST( rb_yield(r) ) )
          $self->erase(current);
      }

      return self;
    }
  }
%enddef

%define %swig_set_methods(set...)

  %swig_sequence_methods_common(%arg(set));
  %swig_sequence_methods_extra_set(%arg(set));

  %fragment("RubyPairBoolOutputIterator","header",fragment=SWIG_From_frag(bool),fragment="RubySequence_Cont") {}

// Redefine std::set iterator/reverse_iterator typemap
%typemap(out,noblock=1) iterator, reverse_iterator {
  $result = SWIG_NewPointerObj(swig::make_set_nonconst_iterator(%static_cast($1,const $type &),
								self),
			          swig::Iterator::descriptor(),SWIG_POINTER_OWN);
 }

// Redefine std::set std::pair<iterator, bool> typemap
  %typemap(out,noblock=1,fragment="RubyPairBoolOutputIterator")
  std::pair<iterator, bool> {
    $result = rb_ary_new2(2);
    rb_ary_push($result, SWIG_NewPointerObj(swig::make_set_nonconst_iterator(%static_cast($1,$type &).first),
                                            swig::Iterator::descriptor(),SWIG_POINTER_OWN));
    rb_ary_push($result, SWIG_From(bool)(%static_cast($1,const $type &).second));
   }

  %extend  {
    %alias push "<<";
    value_type push(const value_type& x) {
      self->insert(x);
      return x;
    }
  
    bool __contains__(const value_type& x) {
      return self->find(x) != self->end();
    }

    value_type __getitem__(difference_type i) const throw (std::out_of_range) {
      return *(swig::cgetpos(self, i));
    }

  };
%enddef


%mixin std::set "Enumerable";



%rename("delete")     std::set::__delete__;
%rename("reject!")    std::set::reject_bang;
%rename("map!")       std::set::map_bang;
%rename("empty?")     std::set::empty;
%rename("include?" )  std::set::__contains__ const;
%rename("has_key?" )  std::set::has_key const;

%alias  std::set::push          "<<";


%include <std/std_set.i>

