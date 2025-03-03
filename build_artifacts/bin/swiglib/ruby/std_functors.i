/**
 * @file   std_functors.i
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
 *         %template< IntSet > std::set< int, swig::BinaryPredicate<int> >;
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

%include <rubystdfunctors.swg>

%fragment("StdFunctors");
