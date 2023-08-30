//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CCMATH_ISNAN
#define BOOST_MATH_CCMATH_ISNAN

#include <cmath>
#include <type_traits>
#include <boost/math/tools/is_constant_evaluated.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include <boost/math/tools/is_standalone.hpp>
#ifndef BOOST_MATH_STANDALONE
#include <boost/config.hpp>
#ifdef BOOST_NO_CXX17_IF_CONSTEXPR
#error "The header <boost/math/norms.hpp> can only be used in C++17 and later."
#endif
#endif

namespace boost::math::ccmath {

template <typename T>
inline constexpr bool isnan BOOST_PREVENT_MACRO_SUBSTITUTION(T x)
{
    if(BOOST_MATH_IS_CONSTANT_EVALUATED(x))
    {
        return x != x;
    }
    else
    {
        using boost::math::isnan;

        if constexpr (!std::is_integral_v<T>)
        {
            return (isnan)(x);
        }
        else
        {
            return (isnan)(static_cast<double>(x));
        }
    }
}

}

#endif // BOOST_MATH_CCMATH_ISNAN
