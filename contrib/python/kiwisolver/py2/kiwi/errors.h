/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2017, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file COPYING.txt, distributed with this software.
|----------------------------------------------------------------------------*/
#pragma once
#include <exception>
#include <string>
#include "constraint.h"
#include "variable.h"


namespace kiwi
{


class UnsatisfiableConstraint : public std::exception
{

public:

	UnsatisfiableConstraint( const Constraint& constraint ) :
		m_constraint( constraint ) {}

	~UnsatisfiableConstraint() throw() {}

	const char* what() const throw()
	{
		return "The constraint can not be satisfied.";
	}

	const Constraint& constraint() const
	{
		return m_constraint;
	}

private:

	Constraint m_constraint;
};


class UnknownConstraint : public std::exception
{

public:

	UnknownConstraint( const Constraint& constraint ) :
		m_constraint( constraint ) {}

	~UnknownConstraint() throw() {}

	const char* what() const throw()
	{
		return "The constraint has not been added to the solver.";
	}

	const Constraint& constraint() const
	{
		return m_constraint;
	}

private:

	Constraint m_constraint;
};


class DuplicateConstraint : public std::exception
{

public:

	DuplicateConstraint( const Constraint& constraint ) :
		m_constraint( constraint ) {}

	~DuplicateConstraint() throw() {}

	const char* what() const throw()
	{
		return "The constraint has already been added to the solver.";
	}

	const Constraint& constraint() const
	{
		return m_constraint;
	}

private:

	Constraint m_constraint;
};


class UnknownEditVariable : public std::exception
{

public:

	UnknownEditVariable( const Variable& variable ) :
		m_variable( variable ) {}

	~UnknownEditVariable() throw() {}

	const char* what() const throw()
	{
		return "The edit variable has not been added to the solver.";
	}

	const Variable& variable() const
	{
		return m_variable;
	}

private:

	Variable m_variable;
};


class DuplicateEditVariable : public std::exception
{

public:

	DuplicateEditVariable( const Variable& variable ) :
		m_variable( variable ) {}

	~DuplicateEditVariable() throw() {}

	const char* what() const throw()
	{
		return "The edit variable has already been added to the solver.";
	}

	const Variable& variable() const
	{
		return m_variable;
	}

private:

	Variable m_variable;
};


class BadRequiredStrength : public std::exception
{

public:

	BadRequiredStrength() {}

	~BadRequiredStrength() throw() {}

	const char* what() const throw()
	{
		return "A required strength cannot be used in this context.";
	}
};


class InternalSolverError : public std::exception
{

public:

	InternalSolverError() : m_msg( "An internal solver error ocurred." ) {}

	InternalSolverError( const char* msg ) : m_msg( msg ) {}

	InternalSolverError( const std::string& msg ) : m_msg( msg ) {}

	~InternalSolverError() throw() {}

	const char* what() const throw()
	{
		return m_msg.c_str();
	}

private:

	std::string m_msg;
};

} // namespace kiwi
