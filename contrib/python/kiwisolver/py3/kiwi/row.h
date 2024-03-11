/*-----------------------------------------------------------------------------
| Copyright (c) 2013-2017, Nucleic Development Team.
|
| Distributed under the terms of the Modified BSD License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
#pragma once
#include "maptype.h"
#include "symbol.h"
#include "util.h"

namespace kiwi
{

namespace impl
{

class Row
{

public:
    using CellMap = MapType<Symbol, double>;

    Row() : Row(0.0) {}

    Row(double constant) : m_constant(constant) {}

    Row(const Row &other) = default;

    ~Row() = default;

    const CellMap &cells() const
    {
        return m_cells;
    }

    double constant() const
    {
        return m_constant;
    }

    /* Add a constant value to the row constant.

	The new value of the constant is returned.

	*/
    double add(double value)
    {
        return m_constant += value;
    }

    /* Insert a symbol into the row with a given coefficient.

	If the symbol already exists in the row, the coefficient will be
	added to the existing coefficient. If the resulting coefficient
	is zero, the symbol will be removed from the row.

	*/
    void insert(const Symbol &symbol, double coefficient = 1.0)
    {
        if (nearZero(m_cells[symbol] += coefficient))
            m_cells.erase(symbol);
    }

    /* Insert a row into this row with a given coefficient.

	The constant and the cells of the other row will be multiplied by
	the coefficient and added to this row. Any cell with a resulting
	coefficient of zero will be removed from the row.

	*/
    void insert(const Row &other, double coefficient = 1.0)
    {
        m_constant += other.m_constant * coefficient;

        for (const auto & cellPair : other.m_cells)
        {
            double coeff = cellPair.second * coefficient;
            if (nearZero(m_cells[cellPair.first] += coeff))
                m_cells.erase(cellPair.first);
        }
    }

    /* Remove the given symbol from the row.

	*/
    void remove(const Symbol &symbol)
    {
        auto it = m_cells.find(symbol);
        if (it != m_cells.end())
            m_cells.erase(it);
    }

    /* Reverse the sign of the constant and all cells in the row.

	*/
    void reverseSign()
    {
        m_constant = -m_constant;
        for (auto &cellPair : m_cells)
            cellPair.second = -cellPair.second;
    }

    /* Solve the row for the given symbol.

	This method assumes the row is of the form a * x + b * y + c = 0
	and (assuming solve for x) will modify the row to represent the
	right hand side of x = -b/a * y - c / a. The target symbol will
	be removed from the row, and the constant and other cells will
	be multiplied by the negative inverse of the target coefficient.

	The given symbol *must* exist in the row.

	*/
    void solveFor(const Symbol &symbol)
    {
        double coeff = -1.0 / m_cells[symbol];
        m_cells.erase(symbol);
        m_constant *= coeff;
        for (auto &cellPair : m_cells)
            cellPair.second *= coeff;
    }

    /* Solve the row for the given symbols.

	This method assumes the row is of the form x = b * y + c and will
	solve the row such that y = x / b - c / b. The rhs symbol will be
	removed from the row, the lhs added, and the result divided by the
	negative inverse of the rhs coefficient.

	The lhs symbol *must not* exist in the row, and the rhs symbol
	*must* exist in the row.

	*/
    void solveFor(const Symbol &lhs, const Symbol &rhs)
    {
        insert(lhs, -1.0);
        solveFor(rhs);
    }

    /* Get the coefficient for the given symbol.

	If the symbol does not exist in the row, zero will be returned.

	*/
    double coefficientFor(const Symbol &symbol) const
    {
        CellMap::const_iterator it = m_cells.find(symbol);
        if (it == m_cells.end())
            return 0.0;
        return it->second;
    }

    /* Substitute a symbol with the data from another row.

	Given a row of the form a * x + b and a substitution of the
	form x = 3 * y + c the row will be updated to reflect the
	expression 3 * a * y + a * c + b.

	If the symbol does not exist in the row, this is a no-op.

	*/
    void substitute(const Symbol &symbol, const Row &row)
    {
        auto it = m_cells.find(symbol);
        if (it != m_cells.end())
        {
            double coefficient = it->second;
            m_cells.erase(it);
            insert(row, coefficient);
        }
    }

private:
    CellMap m_cells;
    double m_constant;
};

} // namespace impl

} // namespace kiwi
