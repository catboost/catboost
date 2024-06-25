/*
 * vbitset.h -- a bitset of variable size.
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


#ifndef PIRE_VBITSET_H
#define PIRE_VBITSET_H


#include <string.h>

#include <contrib/libs/pire/pire/stub/stl.h>

namespace Pire {

#ifdef _DEBUG
#define VBITSET_CHECK_SIZE(x) CheckSize(x)
#else
#define VBITSET_CHECK_SIZE(x) x
#endif

/// A bitset with variable width
class BitSet {
public:
	typedef size_t value_type;
	typedef size_t* pointer;
	typedef size_t& reference;
	typedef const size_t& const_reference;

	class const_iterator;

	BitSet()
		: m_data(1, 1)
	{
	}
	BitSet(size_t size)
		: m_data(RoundUp(size + 1) + 1)
		, m_size(size)
	{
		m_data[RoundDown(size)] |= (1U << Remainder(size));
	}

	void Swap(BitSet& s)
	{
		m_data.swap(s.m_data);
		DoSwap(m_size, s.m_size);
	}

	/// Sets the specified bit to 1.
	void Set(size_t pos) {
		m_data[RoundDown(VBITSET_CHECK_SIZE(pos))] |= (1U << Remainder(pos));
	}

	/// Resets the specified bit to 0.
	void Reset(size_t pos) {
		m_data[RoundDown(VBITSET_CHECK_SIZE(pos))] &= ~(1U << Remainder(pos));
	}

	/// Checks whether the specified bit is set to 1.
	bool Test(size_t pos) const {
		return (m_data[RoundDown(VBITSET_CHECK_SIZE(pos))] & (1U << Remainder(pos))) != 0;
	}

	size_t Size() const {
		return m_size;
	}

	void Resize(size_t newsize)
	{
		m_data.resize(RoundUp(newsize + 1));
		if (Remainder(newsize) && !m_data.empty())
			m_data[m_data.size() - 1] &= ((1U << Remainder(newsize)) - 1); // Clear tail
		m_data[RoundDown(newsize)] |= (1U << Remainder(newsize));
	}

	/// Resets all bits to 0.
	void Clear() { memset(&m_data[0], 0, m_data.size() * sizeof(ContainerType)); }

private:
	typedef unsigned char ContainerType;
	static const size_t ItemSize = sizeof(ContainerType) * 8;
	TVector<ContainerType> m_data;
	size_t m_size;

	static size_t RoundUp(size_t x)   { return x / ItemSize + ((x % ItemSize) ? 1 : 0); }
	static size_t RoundDown(size_t x) { return x / ItemSize; }
	static size_t Remainder(size_t x) { return x % ItemSize; }

#ifdef _DEBUG
	size_t CheckSize(size_t size) const
	{
		if (size < m_size)
			return size;
		else
			throw Error("BitSet: subscript out of range");
	}
#endif
};

}

#endif
