/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2017, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#pragma once
#include <utility>
#include "variable.h"


namespace kiwi
{

class Term
{

public:

	Term( Variable variable, double coefficient = 1.0 ) :
		m_variable( std::move(variable) ), m_coefficient( coefficient ) {}

	// to facilitate efficient map -> vector copies
	Term( const std::pair<const Variable, double>& pair ) :
		m_variable( pair.first ), m_coefficient( pair.second ) {}

	Term(const Term&) = default;

	Term(Term&&) noexcept = default;

	~Term() = default;

	const Variable& variable() const
	{
		return m_variable;
	}

	double coefficient() const
	{
		return m_coefficient;
	}

	double value() const
	{
		return m_coefficient * m_variable.value();
	}

	Term& operator=(const Term&) = default;

	Term& operator=(Term&&) noexcept = default;

private:

	Variable m_variable;
	double m_coefficient;
};

} // namespace kiwi
