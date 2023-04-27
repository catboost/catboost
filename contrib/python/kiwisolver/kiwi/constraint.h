/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2017, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file COPYING.txt, distributed with this software.
|----------------------------------------------------------------------------*/
#pragma once
#include <map>
#include <vector>
#include "expression.h"
#include "shareddata.h"
#include "strength.h"
#include "term.h"
#include "variable.h"


namespace kiwi
{

enum RelationalOperator { OP_LE, OP_GE, OP_EQ };


class Constraint
{

public:

	Constraint() : m_data( 0 ) {}

	Constraint( const Expression& expr,
				RelationalOperator op,
				double strength = strength::required ) :
		m_data( new ConstraintData( expr, op, strength ) ) {}

	Constraint( const Constraint& other, double strength ) :
		m_data( new ConstraintData( other, strength ) ) {}

	~Constraint() {}

	const Expression& expression() const
	{
		return m_data->m_expression;
	}

	RelationalOperator op() const
	{
		return m_data->m_op;
	}

	double strength() const
	{
		return m_data->m_strength;
	}

	bool operator!() const
	{
		return !m_data;
	}

private:

	static Expression reduce( const Expression& expr )
	{
		std::map<Variable, double> vars;
		typedef std::vector<Term>::const_iterator iter_t;
		iter_t end = expr.terms().end();
		for( iter_t it = expr.terms().begin(); it != end; ++it )
			vars[ it->variable() ] += it->coefficient();
		std::vector<Term> terms( vars.begin(), vars.end() );
		return Expression( terms, expr.constant() );
	}

	class ConstraintData : public SharedData
	{

	public:

		ConstraintData( const Expression& expr,
					    RelationalOperator op,
					    double strength ) :
			SharedData(),
			m_expression( reduce( expr ) ),
			m_strength( strength::clip( strength ) ),
			m_op( op ) {}

		ConstraintData( const Constraint& other, double strength ) :
			SharedData(),
			m_expression( other.expression() ),
			m_strength( strength::clip( strength ) ),
			m_op( other.op() ) {}

		~ConstraintData() {}

		Expression m_expression;
		double m_strength;
		RelationalOperator m_op;

	private:

		ConstraintData( const ConstraintData& other );

		ConstraintData& operator=( const ConstraintData& other );
	};

	SharedDataPtr<ConstraintData> m_data;

        friend bool operator<( const Constraint& lhs, const Constraint& rhs )
        {
                return lhs.m_data < rhs.m_data;
        }

        friend bool operator==( const Constraint& lhs, const Constraint& rhs )
        {
                return lhs.m_data == rhs.m_data;
        }

        friend bool operator!=( const Constraint& lhs, const Constraint& rhs )
        {
                return lhs.m_data != rhs.m_data;
        }
};

} // namespace kiwi
