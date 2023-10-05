/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2017, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file COPYING.txt, distributed with this software.
|----------------------------------------------------------------------------*/
#pragma once
#include <iostream>
#include <sstream>
#include <vector>
#include "constraint.h"
#include "solverimpl.h"
#include "term.h"


namespace kiwi
{

namespace impl
{

class DebugHelper
{

public:

	static void dump( const SolverImpl& solver, std::ostream& out )
	{
		out << "Objective" << std::endl;
		out << "---------" << std::endl;
		dump( *solver.m_objective, out );
		out << std::endl;
		out << "Tableau" << std::endl;
		out << "-------" << std::endl;
		dump( solver.m_rows, out );
		out << std::endl;
		out << "Infeasible" << std::endl;
		out << "----------" << std::endl;
		dump( solver.m_infeasible_rows, out );
		out << std::endl;
		out << "Variables" << std::endl;
		out << "---------" << std::endl;
		dump( solver.m_vars, out );
		out << std::endl;
		out << "Edit Variables" << std::endl;
		out << "--------------" << std::endl;
		dump( solver.m_edits, out );
		out << std::endl;
		out << "Constraints" << std::endl;
		out << "-----------" << std::endl;
		dump( solver.m_cns, out );
		out << std::endl;
		out << std::endl;
	}

	static void dump( const SolverImpl::RowMap& rows, std::ostream& out )
	{
		typedef SolverImpl::RowMap::const_iterator iter_t;
		iter_t end = rows.end();
		for( iter_t it = rows.begin(); it != end; ++it )
		{
			dump( it->first, out );
			out << " | ";
			dump( *it->second, out );
		}
	}

	static void dump( const std::vector<Symbol>& symbols, std::ostream& out )
	{
		typedef std::vector<Symbol>::const_iterator iter_t;
		iter_t end = symbols.end();
		for( iter_t it = symbols.begin(); it != end; ++it )
		{
			dump( *it, out );
			out << std::endl;
		}
	}

	static void dump( const SolverImpl::VarMap& vars, std::ostream& out )
	{
		typedef SolverImpl::VarMap::const_iterator iter_t;
		iter_t end = vars.end();
		for( iter_t it = vars.begin(); it != end; ++it )
		{
			out << it->first.name() << " = ";
			dump( it->second, out );
			out << std::endl;
		}
	}

	static void dump( const SolverImpl::CnMap& cns, std::ostream& out )
	{
		typedef SolverImpl::CnMap::const_iterator iter_t;
		iter_t end = cns.end();
		for( iter_t it = cns.begin(); it != end; ++it )
			dump( it->first, out );
	}

	static void dump( const SolverImpl::EditMap& edits, std::ostream& out )
	{
		typedef SolverImpl::EditMap::const_iterator iter_t;
		iter_t end = edits.end();
		for( iter_t it = edits.begin(); it != end; ++it )
			out << it->first.name() << std::endl;
	}

	static void dump( const Row& row, std::ostream& out )
	{
		typedef Row::CellMap::const_iterator iter_t;
		out << row.constant();
		iter_t end = row.cells().end();
		for( iter_t it = row.cells().begin(); it != end; ++it )
		{
			out << " + " << it->second << " * ";
			dump( it->first, out );
		}
		out << std::endl;
	}

	static void dump( const Symbol& symbol, std::ostream& out )
	{
		switch( symbol.type() )
		{
			case Symbol::Invalid:
				out << "i";
				break;
			case Symbol::External:
				out << "v";
				break;
			case Symbol::Slack:
				out << "s";
				break;
			case Symbol::Error:
				out << "e";
				break;
			case Symbol::Dummy:
				out << "d";
				break;
			default:
				break;
		}
		out << symbol.id();
	}

	static void dump( const Constraint& cn, std::ostream& out )
	{
		typedef std::vector<Term>::const_iterator iter_t;
		iter_t begin = cn.expression().terms().begin();
		iter_t end = cn.expression().terms().end();
		for( iter_t it = begin; it != end; ++it )
		{
			out << it->coefficient() << " * ";
			out << it->variable().name() << " + ";
		}
		out << cn.expression().constant();
		switch( cn.op() )
		{
			case OP_LE:
				out << " <= 0 ";
				break;
			case OP_GE:
				out << " >= 0 ";
				break;
			case OP_EQ:
				out << " == 0 ";
				break;
			default:
				break;
		}
		out << " | strength = " << cn.strength() << std::endl;
	}
};

} // namespace impl


namespace debug
{

template<typename T>
void dump( const T& value )
{
	impl::DebugHelper::dump( value, std::cout );
}

template<typename T>
void dump( const T& value, std::ostream& out )
{
	impl::DebugHelper::dump( value, out );
}

template<typename T>
std::string dumps( const T& value )
{
	std::stringstream stream;
	impl::DebugHelper::dump( value, stream );
	return stream.str();
}

} // namespace debug

} // namespace kiwi
