/*
 * static_assert.h -- compile-time assertions
 *
 * Copyright (c) 2007-2010, Dmitry Prokoptsev <dprokoptsev@gmail.com>,
 *                          Alexander Gololobov <agololobov@gmail.com>
 *
 * This file is part of Pire, the Perl Incompatible
 * Regular Expressions library.
 *
 * Pire is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Pire is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser Public License for more details.
 * You should have received a copy of the GNU Lesser Public License
 * along with Pire.  If not, see <http://www.gnu.org/licenses>.
 */

#ifndef PIRE_ASSERT_H_INCLUDED
#define PIRE_ASSERT_H_INCLUDED

namespace Pire { namespace Impl {
        
    // A static (compile-tile) assertion.
    // The idea was shamelessly borrowed from Boost.
    template<bool x> struct StaticAssertion;
    template<> struct StaticAssertion<true> {};
#define PIRE_STATIC_ASSERT(x) \
    enum { PireStaticAssertion ## __LINE__ = sizeof(Pire::Impl::StaticAssertion<(bool) (x)>) }
}}

#endif
