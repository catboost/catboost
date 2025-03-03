/**
 * @file   rubystdfunctors.swg
 * @date   Sun May  6 00:44:33 2007
 * 
 * @brief  This file provides unary and binary functors for STL
 *         containers, that will invoke a Ruby proc or method to do
 *         their operation.
 *
 *         You can use them in a swig file like:
 *
 *         %include <std_set.i>
 *         %include <std_functors.i>
 *
 *         %template< IntSet > std::set< int, swig::BinaryPredicate<> >;
 *
 *
 *         which will then allow calling them from Ruby either like:
 *  
 *            # order of set is defined by C++ default
 *            a = IntSet.new
 *
 *            # sort order defined by Ruby proc
 *            b = IntSet.new( proc { |a,b| a > b } )
 * 
 */

%include rubyclasses.swg


namespace swig {

  %apply GC_VALUE { UnaryPredicate, BinaryPredicate, UnaryFunction,
		      BinaryFunction };

  %typecheck(SWIG_TYPECHECK_POINTER,noblock=1) 
    UnaryPredicate, UnaryPredicate&, UnaryFunction, UnaryFunction&
  {
    $1 = SWIG_Ruby_isCallable($input) && SWIG_Ruby_arity($input, 1);
  }

  %typecheck(SWIG_TYPECHECK_POINTER,noblock=1) 
    BinaryPredicate, BinaryPredicate&, BinaryFunction, BinaryFunction& {
    $1 = SWIG_Ruby_isCallable($input) && SWIG_Ruby_arity($input, 2);
  }

  %typemap(in,noblock=1)  BinaryFunction&, BinaryFunction {
    $1 = new swig::BinaryFunction< >($input);
  }
  %typemap(in,noblock=1) UnaryFunction&, UnaryFunction {
    $1 = new swig::UnaryFunction< >($input);
  }

  %typemap(in,noblock=1) BinaryPredicate&, BinaryPredicate {
    $1 = new swig::BinaryPredicate<>($input);
  }

  %typemap(in,noblock=1) UnaryPredicate&, UnaryPredicate {
    $1 = new swig::UnaryPredicate< >($input);
  }


  %ignore BinaryFunction;
  template< class _T = GC_VALUE >
  struct BinaryFunction {
  };

  %ignore UnaryFunction;
  template< class _T = GC_VALUE >
  struct UnaryFunction {
  };

  %ignore BinaryPredicate;
  template< class _T = GC_VALUE >
  struct BinaryPredicate {
  };

  %ignore UnaryPredicate;
  template< class _T = GC_VALUE >
  struct UnaryPredicate {

  };

}


%fragment("StdFunctors","header",fragment="StdTraits",fragment="GC_VALUE_definition")
{
namespace swig {

  static ID call_id = rb_intern("call");

  template <class _T = GC_VALUE, class _DefaultFunc = std::less<GC_VALUE> >
  struct BinaryPredicate : GC_VALUE
  {
    BinaryPredicate(VALUE obj = Qnil) : GC_VALUE(obj) { }
    bool operator()(_T a, _T b) const
    {
      if (_obj != Qnil) {
        SWIG_RUBY_THREAD_BEGIN_BLOCK;
	VALUE arg1 = swig::from(a);
	VALUE arg2 = swig::from(b);
	VALUE res = rb_funcall( _obj, swig::call_id, 2, arg1, arg2);
        SWIG_RUBY_THREAD_END_BLOCK;
        return RTEST(res);
      } else {
        return _DefaultFunc()(a, b);
      }
    }
  };

  template <class _T = GC_VALUE, class _DefaultFunc = std::less< _T > >
  struct BinaryFunction : GC_VALUE
  {
    BinaryFunction(VALUE obj = Qnil) : GC_VALUE(obj) { }
    _T operator()(_T a, _T b) const
    {
      if (_obj != Qnil) {
        SWIG_RUBY_THREAD_BEGIN_BLOCK;
	VALUE arg1 = swig::from(a);
	VALUE arg2 = swig::from(b);
	VALUE res = rb_funcall( _obj, swig::call_id, 2, arg1, arg2);
        SWIG_RUBY_THREAD_END_BLOCK;
        return swig::as<_T >(res);
      } else {
        return _DefaultFunc()(a, b);
      }
    }
  };

  template< class _T = GC_VALUE >
  struct UnaryPredicate : GC_VALUE
  {
    UnaryPredicate(VALUE obj = Qnil) : GC_VALUE(obj) { }
    bool operator()(_T a) const
    {
      SWIG_RUBY_THREAD_BEGIN_BLOCK;
      VALUE arg1 = swig::from<_T >(a);
      VALUE res = rb_funcall( _obj, swig::call_id, 1, arg1);
      SWIG_RUBY_THREAD_END_BLOCK;
      return RTEST(res);
    }
  };

  template< class _T = GC_VALUE >
  struct UnaryFunction : GC_VALUE
  {
    UnaryFunction(VALUE obj = Qnil) : GC_VALUE(obj) { }
    _T operator()(_T a) const
    {
      SWIG_RUBY_THREAD_BEGIN_BLOCK;
      VALUE arg1 = swig::from(a);
      VALUE res = rb_funcall( _obj, swig::call_id, 1, VALUE(arg1));
      SWIG_RUBY_THREAD_END_BLOCK;
      return swig::as< _T >(res);
    }
  };

} // namespace swig

}


