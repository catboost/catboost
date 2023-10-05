/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2017, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file COPYING.txt, distributed with this software.
|----------------------------------------------------------------------------*/
#pragma once
#include <vector>
#include "constraint.h"
#include "expression.h"
#include "term.h"
#include "variable.h"


namespace kiwi
{

// Variable multiply, divide, and unary invert

inline
Term operator*( const Variable& variable, double coefficient )
{
	return Term( variable, coefficient );
}


inline
Term operator/( const Variable& variable, double denominator )
{
	return variable * ( 1.0 / denominator );
}


inline
Term operator-( const Variable& variable )
{
	return variable * -1.0;
}


// Term multiply, divide, and unary invert

inline
Term operator*( const Term& term, double coefficient )
{
	return Term( term.variable(), term.coefficient() * coefficient );
}


inline
Term operator/( const Term& term, double denominator )
{
	return term * ( 1.0 / denominator );
}


inline
Term operator-( const Term& term )
{
	return term * -1.0;
}


// Expression multiply, divide, and unary invert

inline
Expression operator*( const Expression& expression, double coefficient )
{
	std::vector<Term> terms;
	terms.reserve( expression.terms().size() );
	typedef std::vector<Term>::const_iterator iter_t;
	iter_t begin = expression.terms().begin();
	iter_t end = expression.terms().end();
	for( iter_t it = begin; it != end; ++it )
		terms.push_back( ( *it ) * coefficient );
	return Expression( terms, expression.constant() * coefficient );
}


inline
Expression operator/( const Expression& expression, double denominator )
{
	return expression * ( 1.0 / denominator );
}


inline
Expression operator-( const Expression& expression )
{
	return expression * -1.0;
}


// Double multiply

inline
Expression operator*( double coefficient, const Expression& expression )
{
	return expression * coefficient;
}


inline
Term operator*( double coefficient, const Term& term )
{
	return term * coefficient;
}


inline
Term operator*( double coefficient, const Variable& variable )
{
	return variable * coefficient;
}


// Expression add and subtract

inline
Expression operator+( const Expression& first, const Expression& second )
{
	std::vector<Term> terms;
	terms.reserve( first.terms().size() + second.terms().size() );
	terms.insert( terms.begin(), first.terms().begin(), first.terms().end() );
	terms.insert( terms.end(), second.terms().begin(), second.terms().end() );
	return Expression( terms, first.constant() + second.constant() );
}


inline
Expression operator+( const Expression& first, const Term& second )
{
	std::vector<Term> terms;
	terms.reserve( first.terms().size() + 1 );
	terms.insert( terms.begin(), first.terms().begin(), first.terms().end() );
	terms.push_back( second );
	return Expression( terms, first.constant() );
}


inline
Expression operator+( const Expression& expression, const Variable& variable )
{
	return expression + Term( variable );
}


inline
Expression operator+( const Expression& expression, double constant )
{
	return Expression( expression.terms(), expression.constant() + constant );
}


inline
Expression operator-( const Expression& first, const Expression& second )
{
	return first + -second;
}


inline
Expression operator-( const Expression& expression, const Term& term )
{
	return expression + -term;
}


inline
Expression operator-( const Expression& expression, const Variable& variable )
{
	return expression + -variable;
}


inline
Expression operator-( const Expression& expression, double constant )
{
	return expression + -constant;
}


// Term add and subtract

inline
Expression operator+( const Term& term, const Expression& expression )
{
	return expression + term;
}


inline
Expression operator+( const Term& first, const Term& second )
{
	std::vector<Term> terms;
	terms.reserve( 2 );
	terms.push_back( first );
	terms.push_back( second );
	return Expression( terms );
}


inline
Expression operator+( const Term& term, const Variable& variable )
{
	return term + Term( variable );
}


inline
Expression operator+( const Term& term, double constant )
{
	return Expression( term, constant );
}


inline
Expression operator-( const Term& term, const Expression& expression )
{
	return -expression + term;
}


inline
Expression operator-( const Term& first, const Term& second )
{
	return first + -second;
}


inline
Expression operator-( const Term& term, const Variable& variable )
{
	return term + -variable;
}


inline
Expression operator-( const Term& term, double constant )
{
	return term + -constant;
}


// Variable add and subtract

inline
Expression operator+( const Variable& variable, const Expression& expression )
{
	return expression + variable;
}


inline
Expression operator+( const Variable& variable, const Term& term )
{
	return term + variable;
}


inline
Expression operator+( const Variable& first, const Variable& second )
{
	return Term( first ) + second;
}


inline
Expression operator+( const Variable& variable, double constant )
{
	return Term( variable ) + constant;
}


inline
Expression operator-( const Variable& variable, const Expression& expression )
{
	return variable + -expression;
}


inline
Expression operator-( const Variable& variable, const Term& term )
{
	return variable + -term;
}


inline
Expression operator-( const Variable& first, const Variable& second )
{
	return first + -second;
}


inline
Expression operator-( const Variable& variable, double constant )
{
	return variable + -constant;
}


// Double add and subtract

inline
Expression operator+( double constant, const Expression& expression )
{
	return expression + constant;
}


inline
Expression operator+( double constant, const Term& term )
{
	return term + constant;
}


inline
Expression operator+( double constant, const Variable& variable )
{
	return variable + constant;
}


inline
Expression operator-( double constant, const Expression& expression )
{
	return -expression + constant;
}


inline
Expression operator-( double constant, const Term& term )
{
	return -term + constant;
}


inline
Expression operator-( double constant, const Variable& variable )
{
	return -variable + constant;
}


// Expression relations

inline
Constraint operator==( const Expression& first, const Expression& second )
{
	return Constraint( first - second, OP_EQ );
}


inline
Constraint operator==( const Expression& expression, const Term& term )
{
	return expression == Expression( term );
}


inline
Constraint operator==( const Expression& expression, const Variable& variable )
{
	return expression == Term( variable );
}


inline
Constraint operator==( const Expression& expression, double constant )
{
	return expression == Expression( constant );
}


inline
Constraint operator<=( const Expression& first, const Expression& second )
{
	return Constraint( first - second, OP_LE );
}


inline
Constraint operator<=( const Expression& expression, const Term& term )
{
	return expression <= Expression( term );
}


inline
Constraint operator<=( const Expression& expression, const Variable& variable )
{
	return expression <= Term( variable );
}


inline
Constraint operator<=( const Expression& expression, double constant )
{
	return expression <= Expression( constant );
}


inline
Constraint operator>=( const Expression& first, const Expression& second )
{
	return Constraint( first - second, OP_GE );
}


inline
Constraint operator>=( const Expression& expression, const Term& term )
{
	return expression >= Expression( term );
}


inline
Constraint operator>=( const Expression& expression, const Variable& variable )
{
	return expression >= Term( variable );
}


inline
Constraint operator>=( const Expression& expression, double constant )
{
	return expression >= Expression( constant );
}


// Term relations

inline
Constraint operator==( const Term& term, const Expression& expression )
{
	return expression == term;
}


inline
Constraint operator==( const Term& first, const Term& second )
{
	return Expression( first ) == second;
}


inline
Constraint operator==( const Term& term, const Variable& variable )
{
	return Expression( term ) == variable;
}


inline
Constraint operator==( const Term& term, double constant )
{
	return Expression( term ) == constant;
}


inline
Constraint operator<=( const Term& term, const Expression& expression )
{
	return expression >= term;
}


inline
Constraint operator<=( const Term& first, const Term& second )
{
	return Expression( first ) <= second;
}


inline
Constraint operator<=( const Term& term, const Variable& variable )
{
	return Expression( term ) <= variable;
}


inline
Constraint operator<=( const Term& term, double constant )
{
	return Expression( term ) <= constant;
}


inline
Constraint operator>=( const Term& term, const Expression& expression )
{
	return expression <= term;
}


inline
Constraint operator>=( const Term& first, const Term& second )
{
	return Expression( first ) >= second;
}


inline
Constraint operator>=( const Term& term, const Variable& variable )
{
	return Expression( term ) >= variable;
}


inline
Constraint operator>=( const Term& term, double constant )
{
	return Expression( term ) >= constant;
}


// Variable relations
inline
Constraint operator==( const Variable& variable, const Expression& expression )
{
	return expression == variable;
}


inline
Constraint operator==( const Variable& variable, const Term& term )
{
	return term == variable;
}


inline
Constraint operator==( const Variable& first, const Variable& second )
{
	return Term( first ) == second;
}


inline
Constraint operator==( const Variable& variable, double constant )
{
	return Term( variable ) == constant;
}


inline
Constraint operator<=( const Variable& variable, const Expression& expression )
{
	return expression >= variable;
}


inline
Constraint operator<=( const Variable& variable, const Term& term )
{
	return term >= variable;
}


inline
Constraint operator<=( const Variable& first, const Variable& second )
{
	return Term( first ) <= second;
}


inline
Constraint operator<=( const Variable& variable, double constant )
{
	return Term( variable ) <= constant;
}


inline
Constraint operator>=( const Variable& variable, const Expression& expression )
{
	return expression <= variable;
}


inline
Constraint operator>=( const Variable& variable, const Term& term )
{
	return term <= variable;
}


inline
Constraint operator>=( const Variable& first, const Variable& second )
{
	return Term( first ) >= second;
}


inline
Constraint operator>=( const Variable& variable, double constant )
{
	return Term( variable ) >= constant;
}


// Double relations

inline
Constraint operator==( double constant, const Expression& expression )
{
	return expression == constant;
}


inline
Constraint operator==( double constant, const Term& term )
{
	return term == constant;
}


inline
Constraint operator==( double constant, const Variable& variable )
{
	return variable == constant;
}


inline
Constraint operator<=( double constant, const Expression& expression )
{
	return expression >= constant;
}


inline
Constraint operator<=( double constant, const Term& term )
{
	return term >= constant;
}


inline
Constraint operator<=( double constant, const Variable& variable )
{
	return variable >= constant;
}


inline
Constraint operator>=( double constant, const Expression& expression )
{
	return expression <= constant;
}


inline
Constraint operator>=( double constant, const Term& term )
{
	return term <= constant;
}


inline
Constraint operator>=( double constant, const Variable& variable )
{
	return variable <= constant;
}


// Constraint strength modifier

inline
Constraint operator|( const Constraint& constraint, double strength )
{
	return Constraint( constraint, strength );
}


inline
Constraint operator|( double strength, const Constraint& constraint )
{
	return constraint | strength;
}

} // namespace kiwi
