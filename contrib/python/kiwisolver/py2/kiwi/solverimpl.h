/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2017, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file COPYING.txt, distributed with this software.
|----------------------------------------------------------------------------*/
#pragma once
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>
#include "constraint.h"
#include "errors.h"
#include "expression.h"
#include "maptype.h"
#include "row.h"
#include "symbol.h"
#include "term.h"
#include "util.h"
#include "variable.h"


namespace kiwi
{

namespace impl
{

class SolverImpl
{
	friend class DebugHelper;

	struct Tag
	{
		Symbol marker;
		Symbol other;
	};

	struct EditInfo
	{
		Tag tag;
		Constraint constraint;
		double constant;
	};

	typedef MapType<Variable, Symbol>::Type VarMap;

	typedef MapType<Symbol, Row*>::Type RowMap;

	typedef MapType<Constraint, Tag>::Type CnMap;

	typedef MapType<Variable, EditInfo>::Type EditMap;

	struct DualOptimizeGuard
	{
		DualOptimizeGuard( SolverImpl& impl ) : m_impl( impl ) {}
		~DualOptimizeGuard() { m_impl.dualOptimize(); }
		SolverImpl& m_impl;
	};

public:

	SolverImpl() : m_objective( new Row() ), m_id_tick( 1 ) {}

	~SolverImpl() { clearRows(); }

	/* Add a constraint to the solver.

	Throws
	------
	DuplicateConstraint
		The given constraint has already been added to the solver.

	UnsatisfiableConstraint
		The given constraint is required and cannot be satisfied.

	*/
	void addConstraint( const Constraint& constraint )
	{
		if( m_cns.find( constraint ) != m_cns.end() )
			throw DuplicateConstraint( constraint );

		// Creating a row causes symbols to be reserved for the variables
		// in the constraint. If this method exits with an exception,
		// then its possible those variables will linger in the var map.
		// Since its likely that those variables will be used in other
		// constraints and since exceptional conditions are uncommon,
		// i'm not too worried about aggressive cleanup of the var map.
		Tag tag;
		std::auto_ptr<Row> rowptr( createRow( constraint, tag ) );
		Symbol subject( chooseSubject( *rowptr, tag ) );

		// If chooseSubject could not find a valid entering symbol, one
		// last option is available if the entire row is composed of
		// dummy variables. If the constant of the row is zero, then
		// this represents redundant constraints and the new dummy
		// marker can enter the basis. If the constant is non-zero,
		// then it represents an unsatisfiable constraint.
		if( subject.type() == Symbol::Invalid && allDummies( *rowptr ) )
		{
			if( !nearZero( rowptr->constant() ) )
				throw UnsatisfiableConstraint( constraint );
			else
				subject = tag.marker;
		}

		// If an entering symbol still isn't found, then the row must
		// be added using an artificial variable. If that fails, then
		// the row represents an unsatisfiable constraint.
		if( subject.type() == Symbol::Invalid )
		{
			if( !addWithArtificialVariable( *rowptr ) )
				throw UnsatisfiableConstraint( constraint );
		}
		else
		{
			rowptr->solveFor( subject );
			substitute( subject, *rowptr );
			m_rows[ subject ] = rowptr.release();
		}

		m_cns[ constraint ] = tag;

		// Optimizing after each constraint is added performs less
		// aggregate work due to a smaller average system size. It
		// also ensures the solver remains in a consistent state.
		optimize( *m_objective );
	}

	/* Remove a constraint from the solver.

	Throws
	------
	UnknownConstraint
		The given constraint has not been added to the solver.

	*/
	void removeConstraint( const Constraint& constraint )
	{
		CnMap::iterator cn_it = m_cns.find( constraint );
		if( cn_it == m_cns.end() )
			throw UnknownConstraint( constraint );

		Tag tag( cn_it->second );
		m_cns.erase( cn_it );

		// Remove the error effects from the objective function
		// *before* pivoting, or substitutions into the objective
		// will lead to incorrect solver results.
		removeConstraintEffects( constraint, tag );

		// If the marker is basic, simply drop the row. Otherwise,
		// pivot the marker into the basis and then drop the row.
		RowMap::iterator row_it = m_rows.find( tag.marker );
		if( row_it != m_rows.end() )
		{
			std::auto_ptr<Row> rowptr( row_it->second );
			m_rows.erase( row_it );
		}
		else
		{
			row_it = getMarkerLeavingRow( tag.marker );
			if( row_it == m_rows.end() )
				throw InternalSolverError( "failed to find leaving row" );
			Symbol leaving( row_it->first );
			std::auto_ptr<Row> rowptr( row_it->second );
			m_rows.erase( row_it );
			rowptr->solveFor( leaving, tag.marker );
			substitute( tag.marker, *rowptr );
		}

		// Optimizing after each constraint is removed ensures that the
		// solver remains consistent. It makes the solver api easier to
		// use at a small tradeoff for speed.
		optimize( *m_objective );
	}

	/* Test whether a constraint has been added to the solver.

	*/
	bool hasConstraint( const Constraint& constraint ) const
	{
		return m_cns.find( constraint ) != m_cns.end();
	}

	/* Add an edit variable to the solver.

	This method should be called before the `suggestValue` method is
	used to supply a suggested value for the given edit variable.

	Throws
	------
	DuplicateEditVariable
		The given edit variable has already been added to the solver.

	BadRequiredStrength
		The given strength is >= required.

	*/
	void addEditVariable( const Variable& variable, double strength )
	{
		if( m_edits.find( variable ) != m_edits.end() )
			throw DuplicateEditVariable( variable );
		strength = strength::clip( strength );
		if( strength == strength::required )
			throw BadRequiredStrength();
		Constraint cn( Expression( variable ), OP_EQ, strength );
		addConstraint( cn );
		EditInfo info;
		info.tag = m_cns[ cn ];
		info.constraint = cn;
		info.constant = 0.0;
		m_edits[ variable ] = info;
	}

	/* Remove an edit variable from the solver.

	Throws
	------
	UnknownEditVariable
		The given edit variable has not been added to the solver.

	*/
	void removeEditVariable( const Variable& variable )
	{
		EditMap::iterator it = m_edits.find( variable );
		if( it == m_edits.end() )
			throw UnknownEditVariable( variable );
		removeConstraint( it->second.constraint );
		m_edits.erase( it );
	}

	/* Test whether an edit variable has been added to the solver.

	*/
	bool hasEditVariable( const Variable& variable ) const
	{
		return m_edits.find( variable ) != m_edits.end();
	}

	/* Suggest a value for the given edit variable.

	This method should be used after an edit variable as been added to
	the solver in order to suggest the value for that variable.

	Throws
	------
	UnknownEditVariable
		The given edit variable has not been added to the solver.

	*/
	void suggestValue( const Variable& variable, double value )
	{
		EditMap::iterator it = m_edits.find( variable );
		if( it == m_edits.end() )
			throw UnknownEditVariable( variable );

		DualOptimizeGuard guard( *this );
		EditInfo& info = it->second;
		double delta = value - info.constant;
		info.constant = value;

		// Check first if the positive error variable is basic.
		RowMap::iterator row_it = m_rows.find( info.tag.marker );
		if( row_it != m_rows.end() )
		{
			if( row_it->second->add( -delta ) < 0.0 )
				m_infeasible_rows.push_back( row_it->first );
			return;
		}

		// Check next if the negative error variable is basic.
		row_it = m_rows.find( info.tag.other );
		if( row_it != m_rows.end() )
		{
			if( row_it->second->add( delta ) < 0.0 )
				m_infeasible_rows.push_back( row_it->first );
			return;
		}

		// Otherwise update each row where the error variables exist.
		RowMap::iterator end = m_rows.end();
		for( row_it = m_rows.begin(); row_it != end; ++row_it )
		{
			double coeff = row_it->second->coefficientFor( info.tag.marker );
			if( coeff != 0.0 &&
				row_it->second->add( delta * coeff ) < 0.0 &&
				row_it->first.type() != Symbol::External )
				m_infeasible_rows.push_back( row_it->first );
		}
	}

	/* Update the values of the external solver variables.

	*/
	void updateVariables()
	{
		typedef RowMap::iterator row_iter_t;
		typedef VarMap::iterator var_iter_t;
		row_iter_t row_end = m_rows.end();
		var_iter_t var_end = m_vars.end();
		for( var_iter_t var_it = m_vars.begin(); var_it != var_end; ++var_it )
		{
			Variable& var( const_cast<Variable&>( var_it->first ) );
			row_iter_t row_it = m_rows.find( var_it->second );
			if( row_it == row_end )
				var.setValue( 0.0 );
			else
				var.setValue( row_it->second->constant() );
		}
	}

	/* Reset the solver to the empty starting condition.

	This method resets the internal solver state to the empty starting
	condition, as if no constraints or edit variables have been added.
	This can be faster than deleting the solver and creating a new one
	when the entire system must change, since it can avoid unecessary
	heap (de)allocations.

	*/
	void reset()
	{
		clearRows();
		m_cns.clear();
		m_vars.clear();
		m_edits.clear();
		m_infeasible_rows.clear();
		m_objective.reset( new Row() );
		m_artificial.reset();
		m_id_tick = 1;
	}

private:

	SolverImpl( const SolverImpl& );

	SolverImpl& operator=( const SolverImpl& );

	struct RowDeleter
	{
		template<typename T>
		void operator()( T& pair ) { delete pair.second; }
	};

	void clearRows()
	{
		std::for_each( m_rows.begin(), m_rows.end(), RowDeleter() );
		m_rows.clear();
	}

	/* Get the symbol for the given variable.

	If a symbol does not exist for the variable, one will be created.

	*/
	Symbol getVarSymbol( const Variable& variable )
	{
		VarMap::iterator it = m_vars.find( variable );
		if( it != m_vars.end() )
			return it->second;
		Symbol symbol( Symbol::External, m_id_tick++ );
		m_vars[ variable ] = symbol;
		return symbol;
	}

	/* Create a new Row object for the given constraint.

	The terms in the constraint will be converted to cells in the row.
	Any term in the constraint with a coefficient of zero is ignored.
	This method uses the `getVarSymbol` method to get the symbol for
	the variables added to the row. If the symbol for a given cell
	variable is basic, the cell variable will be substituted with the
	basic row.

	The necessary slack and error variables will be added to the row.
	If the constant for the row is negative, the sign for the row
	will be inverted so the constant becomes positive.

	The tag will be updated with the marker and error symbols to use
	for tracking the movement of the constraint in the tableau.

	*/
	Row* createRow( const Constraint& constraint, Tag& tag )
	{
		typedef std::vector<Term>::const_iterator iter_t;
		const Expression& expr( constraint.expression() );
		Row* row = new Row( expr.constant() );

		// Substitute the current basic variables into the row.
		iter_t end = expr.terms().end();
		for( iter_t it = expr.terms().begin(); it != end; ++it )
		{
			if( !nearZero( it->coefficient() ) )
			{
				Symbol symbol( getVarSymbol( it->variable() ) );
				RowMap::const_iterator row_it = m_rows.find( symbol );
				if( row_it != m_rows.end() )
					row->insert( *row_it->second, it->coefficient() );
				else
					row->insert( symbol, it->coefficient() );
			}
		}

		// Add the necessary slack, error, and dummy variables.
		switch( constraint.op() )
		{
			case OP_LE:
			case OP_GE:
			{
				double coeff = constraint.op() == OP_LE ? 1.0 : -1.0;
				Symbol slack( Symbol::Slack, m_id_tick++ );
				tag.marker = slack;
				row->insert( slack, coeff );
				if( constraint.strength() < strength::required )
				{
					Symbol error( Symbol::Error, m_id_tick++ );
					tag.other = error;
					row->insert( error, -coeff );
					m_objective->insert( error, constraint.strength() );
				}
				break;
			}
			case OP_EQ:
			{
				if( constraint.strength() < strength::required )
				{
					Symbol errplus( Symbol::Error, m_id_tick++ );
					Symbol errminus( Symbol::Error, m_id_tick++ );
					tag.marker = errplus;
					tag.other = errminus;
					row->insert( errplus, -1.0 ); // v = eplus - eminus
					row->insert( errminus, 1.0 ); // v - eplus + eminus = 0
					m_objective->insert( errplus, constraint.strength() );
					m_objective->insert( errminus, constraint.strength() );
				}
				else
				{
					Symbol dummy( Symbol::Dummy, m_id_tick++ );
					tag.marker = dummy;
					row->insert( dummy );
				}
				break;
			}
		}

		// Ensure the row as a positive constant.
		if( row->constant() < 0.0 )
			row->reverseSign();

		return row;
	}

	/* Choose the subject for solving for the row.

	This method will choose the best subject for using as the solve
	target for the row. An invalid symbol will be returned if there
	is no valid target.

	The symbols are chosen according to the following precedence:

	1) The first symbol representing an external variable.
	2) A negative slack or error tag variable.

	If a subject cannot be found, an invalid symbol will be returned.

	*/
	Symbol chooseSubject( const Row& row, const Tag& tag )
	{
		typedef Row::CellMap::const_iterator iter_t;
		iter_t end = row.cells().end();
		for( iter_t it = row.cells().begin(); it != end; ++it )
		{
			if( it->first.type() == Symbol::External )
				return it->first;
		}
		if( tag.marker.type() == Symbol::Slack || tag.marker.type() == Symbol::Error )
		{
			if( row.coefficientFor( tag.marker ) < 0.0 )
				return tag.marker;
		}
		if( tag.other.type() == Symbol::Slack || tag.other.type() == Symbol::Error )
		{
			if( row.coefficientFor( tag.other ) < 0.0 )
				return tag.other;
		}
		return Symbol();
	}

 	/* Add the row to the tableau using an artificial variable.

	This will return false if the constraint cannot be satisfied.

 	*/
 	bool addWithArtificialVariable( const Row& row )
 	{
		// Create and add the artificial variable to the tableau
		Symbol art( Symbol::Slack, m_id_tick++ );
		m_rows[ art ] = new Row( row );
		m_artificial.reset( new Row( row ) );

		// Optimize the artificial objective. This is successful
		// only if the artificial objective is optimized to zero.
		optimize( *m_artificial );
		bool success = nearZero( m_artificial->constant() );
		m_artificial.reset();

		// If the artificial variable is not basic, pivot the row so that
		// it becomes basic. If the row is constant, exit early.
		RowMap::iterator it = m_rows.find( art );
		if( it != m_rows.end() )
		{
			std::auto_ptr<Row> rowptr( it->second );
			m_rows.erase( it );
			if( rowptr->cells().empty() )
				return success;
			Symbol entering( anyPivotableSymbol( *rowptr ) );
			if( entering.type() == Symbol::Invalid )
				return false;  // unsatisfiable (will this ever happen?)
			rowptr->solveFor( art, entering );
			substitute( entering, *rowptr );
			m_rows[ entering ] = rowptr.release();
		}

		// Remove the artificial variable from the tableau.
		RowMap::iterator end = m_rows.end();
		for( it = m_rows.begin(); it != end; ++it )
			it->second->remove( art );
		m_objective->remove( art );
		return success;
 	}

	/* Substitute the parametric symbol with the given row.

	This method will substitute all instances of the parametric symbol
	in the tableau and the objective function with the given row.

	*/
	void substitute( const Symbol& symbol, const Row& row )
	{
		typedef RowMap::iterator iter_t;
		iter_t end = m_rows.end();
		for( iter_t it = m_rows.begin(); it != end; ++it )
		{
			it->second->substitute( symbol, row );
			if( it->first.type() != Symbol::External &&
				it->second->constant() < 0.0 )
				m_infeasible_rows.push_back( it->first );
		}
		m_objective->substitute( symbol, row );
		if( m_artificial.get() )
			m_artificial->substitute( symbol, row );
	}

	/* Optimize the system for the given objective function.

	This method performs iterations of Phase 2 of the simplex method
	until the objective function reaches a minimum.

	Throws
	------
	InternalSolverError
		The value of the objective function is unbounded.

	*/
	void optimize( const Row& objective )
	{
		while( true )
		{
			Symbol entering( getEnteringSymbol( objective ) );
			if( entering.type() == Symbol::Invalid )
				return;
			RowMap::iterator it = getLeavingRow( entering );
			if( it == m_rows.end() )
				throw InternalSolverError( "The objective is unbounded." );
			// pivot the entering symbol into the basis
			Symbol leaving( it->first );
			Row* row = it->second;
			m_rows.erase( it );
			row->solveFor( leaving, entering );
			substitute( entering, *row );
			m_rows[ entering ] = row;
		}
	}

	/* Optimize the system using the dual of the simplex method.

	The current state of the system should be such that the objective
	function is optimal, but not feasible. This method will perform
	an iteration of the dual simplex method to make the solution both
	optimal and feasible.

	Throws
	------
	InternalSolverError
		The system cannot be dual optimized.

	*/
	void dualOptimize()
	{
		while( !m_infeasible_rows.empty() )
		{

			Symbol leaving( m_infeasible_rows.back() );
			m_infeasible_rows.pop_back();
			RowMap::iterator it = m_rows.find( leaving );
			if( it != m_rows.end() && !nearZero( it->second->constant() ) &&
				it->second->constant() < 0.0 )
			{
				Symbol entering( getDualEnteringSymbol( *it->second ) );
				if( entering.type() == Symbol::Invalid )
					throw InternalSolverError( "Dual optimize failed." );
				// pivot the entering symbol into the basis
				Row* row = it->second;
				m_rows.erase( it );
				row->solveFor( leaving, entering );
				substitute( entering, *row );
				m_rows[ entering ] = row;
			}
		}
	}

	/* Compute the entering variable for a pivot operation.

	This method will return first symbol in the objective function which
	is non-dummy and has a coefficient less than zero. If no symbol meets
	the criteria, it means the objective function is at a minimum, and an
	invalid symbol is returned.

	*/
	Symbol getEnteringSymbol( const Row& objective )
	{
		typedef Row::CellMap::const_iterator iter_t;
		iter_t end = objective.cells().end();
		for( iter_t it = objective.cells().begin(); it != end; ++it )
		{
			if( it->first.type() != Symbol::Dummy && it->second < 0.0 )
				return it->first;
		}
		return Symbol();
	}

	/* Compute the entering symbol for the dual optimize operation.

	This method will return the symbol in the row which has a positive
	coefficient and yields the minimum ratio for its respective symbol
	in the objective function. The provided row *must* be infeasible.
	If no symbol is found which meats the criteria, an invalid symbol
	is returned.

	*/
	Symbol getDualEnteringSymbol( const Row& row )
	{
		typedef Row::CellMap::const_iterator iter_t;
		Symbol entering;
		double ratio = std::numeric_limits<double>::max();
		iter_t end = row.cells().end();
		for( iter_t it = row.cells().begin(); it != end; ++it )
		{
			if( it->second > 0.0 && it->first.type() != Symbol::Dummy )
			{
				double coeff = m_objective->coefficientFor( it->first );
				double r = coeff / it->second;
				if( r < ratio )
				{
					ratio = r;
					entering = it->first;
				}
			}
		}
		return entering;
	}

	/* Get the first Slack or Error symbol in the row.

	If no such symbol is present, and Invalid symbol will be returned.

	*/
	Symbol anyPivotableSymbol( const Row& row )
	{
		typedef Row::CellMap::const_iterator iter_t;
		iter_t end = row.cells().end();
		for( iter_t it = row.cells().begin(); it != end; ++it )
		{
			const Symbol& sym( it->first );
			if( sym.type() == Symbol::Slack || sym.type() == Symbol::Error )
				return sym;
		}
		return Symbol();
	}

	/* Compute the row which holds the exit symbol for a pivot.

	This method will return an iterator to the row in the row map
	which holds the exit symbol. If no appropriate exit symbol is
	found, the end() iterator will be returned. This indicates that
	the objective function is unbounded.

	*/
	RowMap::iterator getLeavingRow( const Symbol& entering )
	{
		typedef RowMap::iterator iter_t;
		double ratio = std::numeric_limits<double>::max();
		iter_t end = m_rows.end();
		iter_t found = m_rows.end();
		for( iter_t it = m_rows.begin(); it != end; ++it )
		{
			if( it->first.type() != Symbol::External )
			{
				double temp = it->second->coefficientFor( entering );
				if( temp < 0.0 )
				{
					double temp_ratio = -it->second->constant() / temp;
					if( temp_ratio < ratio )
					{
						ratio = temp_ratio;
						found = it;
					}
				}
			}
		}
		return found;
	}

	/* Compute the leaving row for a marker variable.

	This method will return an iterator to the row in the row map
	which holds the given marker variable. The row will be chosen
	according to the following precedence:

	1) The row with a restricted basic varible and a negative coefficient
	   for the marker with the smallest ratio of -constant / coefficient.

	2) The row with a restricted basic variable and the smallest ratio
	   of constant / coefficient.

	3) The last unrestricted row which contains the marker.

	If the marker does not exist in any row, the row map end() iterator
	will be returned. This indicates an internal solver error since
	the marker *should* exist somewhere in the tableau.

	*/
	RowMap::iterator getMarkerLeavingRow( const Symbol& marker )
	{
		const double dmax = std::numeric_limits<double>::max();
		typedef RowMap::iterator iter_t;
		double r1 = dmax;
		double r2 = dmax;
		iter_t end = m_rows.end();
		iter_t first = end;
		iter_t second = end;
		iter_t third = end;
		for( iter_t it = m_rows.begin(); it != end; ++it )
		{
			double c = it->second->coefficientFor( marker );
			if( c == 0.0 )
				continue;
			if( it->first.type() == Symbol::External )
			{
				third = it;
			}
			else if( c < 0.0 )
			{
				double r = -it->second->constant() / c;
				if( r < r1 )
				{
					r1 = r;
					first = it;
				}
			}
			else
			{
				double r = it->second->constant() / c;
				if( r < r2 )
				{
					r2 = r;
					second = it;
				}
			}
		}
		if( first != end )
			return first;
		if( second != end )
			return second;
		return third;
	}

	/* Remove the effects of a constraint on the objective function.

	*/
	void removeConstraintEffects( const Constraint& cn, const Tag& tag )
	{
		if( tag.marker.type() == Symbol::Error )
			removeMarkerEffects( tag.marker, cn.strength() );
		if( tag.other.type() == Symbol::Error )
			removeMarkerEffects( tag.other, cn.strength() );
	}

	/* Remove the effects of an error marker on the objective function.

	*/
	void removeMarkerEffects( const Symbol& marker, double strength )
	{
		RowMap::iterator row_it = m_rows.find( marker );
		if( row_it != m_rows.end() )
			m_objective->insert( *row_it->second, -strength );
		else
			m_objective->insert( marker, -strength );
	}

	/* Test whether a row is composed of all dummy variables.

	*/
	bool allDummies( const Row& row )
	{
		typedef Row::CellMap::const_iterator iter_t;
		iter_t end = row.cells().end();
		for( iter_t it = row.cells().begin(); it != end; ++it )
		{
			if( it->first.type() != Symbol::Dummy )
				return false;
		}
		return true;
	}

	CnMap m_cns;
	RowMap m_rows;
	VarMap m_vars;
	EditMap m_edits;
	std::vector<Symbol> m_infeasible_rows;
	std::auto_ptr<Row> m_objective;
	std::auto_ptr<Row> m_artificial;
	Symbol::Id m_id_tick;
};

} // namespace impl

} // namespace kiwi
