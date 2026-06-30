/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/functional.h>
#include <thrust/system/detail/error_condition.inl>

THRUST_NAMESPACE_BEGIN

namespace system
{

error_condition ::error_condition()
    : m_val(0)
    , m_cat(&generic_category())
{
  ;
} // end error_condition::error_condition()

error_condition ::error_condition(int val, const error_category& cat)
    : m_val(val)
    , m_cat(&cat)
{
  ;
} // end error_condition::error_condition()

template <typename ErrorConditionEnum>
error_condition ::error_condition(ErrorConditionEnum e
// XXX WAR msvc's problem with enable_if
#if !_CCCL_COMPILER(MSVC)
                                  ,
                                  ::cuda::std::enable_if_t<is_error_condition_enum<ErrorConditionEnum>::value>*
#endif // !_CCCL_COMPILER(MSVC)
)
{
  *this = make_error_condition(e);
} // end error_condition::error_condition()

void error_condition ::assign(int val, const error_category& cat)
{
  m_val = val;
  m_cat = &cat;
} // end error_category::assign()

template <typename ErrorConditionEnum>
// XXX WAR msvc's problem with enable_if
#if !_CCCL_COMPILER(MSVC)
::cuda::std::enable_if_t<is_error_condition_enum<ErrorConditionEnum>::value, error_condition>&
#else
error_condition&
#endif // !_CCCL_COMPILER(MSVC)
error_condition ::operator=(ErrorConditionEnum e)
{
  *this = make_error_condition(e);
  return *this;
} // end error_condition::operator=()

void error_condition ::clear()
{
  m_val = 0;
  m_cat = &generic_category();
} // end error_condition::clear()

int error_condition ::value() const
{
  return m_val;
} // end error_condition::value()

const error_category& error_condition ::category() const
{
  return *m_cat;
} // end error_condition::category()

std::string error_condition ::message() const
{
  return category().message(value());
} // end error_condition::message()

error_condition ::operator bool() const
{
  return value() != 0;
} // end error_condition::operator bool ()

error_condition make_error_condition(errc::errc_t e)
{
  return error_condition(static_cast<int>(e), generic_category());
} // end make_error_condition()

bool operator<(const error_condition& lhs, const error_condition& rhs)
{
  return lhs.category().operator<(rhs.category())
      || (lhs.category().operator==(rhs.category()) && (lhs.value() < rhs.value()));
} // end operator<()

} // namespace system

THRUST_NAMESPACE_END
