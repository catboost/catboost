#ifndef BOOST_HASH_IS_TUPLE_LIKE_HPP_INCLUDED
#define BOOST_HASH_IS_TUPLE_LIKE_HPP_INCLUDED

// Copyright 2017, 2022 Peter Dimov.
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/type_traits/integral_constant.hpp>
#include <boost/config.hpp>
#include <boost/config/workaround.hpp>
#include <utility>

namespace boost
{
namespace hash_detail
{

template<class T, class E = true_type> struct is_tuple_like_: false_type
{
};

#if !defined(BOOST_NO_CXX11_HDR_TUPLE) && !BOOST_WORKAROUND(BOOST_MSVC, <= 1800)

template<class T> struct is_tuple_like_<T, integral_constant<bool, std::tuple_size<T>::value == std::tuple_size<T>::value> >: true_type
{
};

#endif

} // namespace hash_detail

namespace container_hash
{

template<class T> struct is_tuple_like: hash_detail::is_tuple_like_<T>
{
};

} // namespace container_hash
} // namespace boost

#endif // #ifndef BOOST_HASH_IS_TUPLE_LIKE_HPP_INCLUDED
