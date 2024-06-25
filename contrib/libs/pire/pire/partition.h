/*
 * partition.h -- a disjoint set of pairwise equivalent items
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


#ifndef PIRE_PARTITION_H
#define PIRE_PARTITION_H


#include <contrib/libs/pire/pire/stub/stl.h>
#include <contrib/libs/pire/pire/stub/singleton.h>

namespace Pire {

/*
* A class which forms a disjoint set of pairwise equivalent items,
* depending on given equivalence relation.
*/
template<class T, class Eq>
class Partition {
private:
	typedef TMap< T, ypair< size_t, TVector<T> > > Set;

public:
	Partition(const Eq& eq)
		: m_eq(eq)
		, m_maxidx(0)
	{
	}

	/// Appends a new item into partition, creating new equivalience class if neccessary.
	void Append(const T& t) {
		DoAppend(m_set, t);
	}

	typedef typename Set::const_iterator ConstIterator;

	ConstIterator Begin() const {
		return m_set.begin();
	}
	ConstIterator begin() const {
		return m_set.begin();
	}
	ConstIterator End() const {
		return m_set.end();
	}
	ConstIterator end() const {
		return m_set.end();
	}
	size_t Size() const {
		return m_set.size();
	}
	bool Empty() const {
		return m_set.empty();
	}

	/// Returns an item equal to @p t. It is guaranteed that:
	/// - representative(a) equals representative(b) iff a is equivalent to b;
	/// - representative(a) is equivalent to a.
	const T& Representative(const T& t) const
	{
		auto it = m_inv.find(t);
		if (it != m_inv.end())
			return it->second;
		else
			return DefaultValue<T>();
	}
	
	bool Contains(const T& t) const
	{
		return m_inv.find(t) != m_inv.end();
	}

	/// Returns an index of set containing @p t. It is guaranteed that:
	/// - index(a) equals index(b) iff a is equivalent to b;
	/// - 0 <= index(a) < size().
	size_t Index(const T& t) const
	{
		auto it = m_inv.find(t);
		if (it == m_inv.end())
			throw Error("Partition::index(): attempted to obtain an index of nonexistent item");
		auto it2 = m_set.find(it->second);
		Y_ASSERT(it2 != m_set.end());
		return it2->second.first;
	}
	/// Returns the whole equivalence class of @p t (i.e. item @p i
	/// is returned iff representative(i) == representative(t)).
	const TVector<T>& Klass(const T& t) const
	{
		auto it = m_inv.find(t);
		if (it == m_inv.end())
			throw Error("Partition::index(): attempted to obtain an index of nonexistent item");
		auto it2 = m_set.find(it->second);
		Y_ASSERT(it2 != m_set.end());
		return it2->second.second;
	}

	bool operator == (const Partition& rhs) const { return m_set == rhs.m_set; }
	bool operator != (const Partition& rhs) const { return !(*this == rhs); }

	/// Splits the current sets into smaller ones, using given equivalence relation.
	/// Requires given relation imply previous one (set either in ctor or
	/// in preceeding calls to split()), but performs faster.
	/// Replaces previous relation with given one.
	void Split(const Eq& eq)
	{
		m_eq = eq;

		for (auto&& element : m_set)
			if (element.second.second.size() > 1) {
				TVector<T>& v = element.second.second;
				auto bound = std::partition(v.begin(), v.end(), std::bind2nd(m_eq, v[0]));
				if (bound == v.end())
					continue;

				Set delta;
				for (auto it = bound, ie = v.end(); it != ie; ++it)
					DoAppend(delta, *it);

				v.erase(bound, v.end());
				m_set.insert(delta.begin(), delta.end());
			}
	}

private:
	Eq m_eq;
	Set m_set;
	TMap<T, T> m_inv;
	size_t m_maxidx;

	void DoAppend(Set& set, const T& t)
	{
		auto it = set.begin();
		auto end = set.end();
		for (; it != end; ++it)
			if (m_eq(it->first, t)) {
				it->second.second.push_back(t);
				m_inv[t] = it->first;
				break;
			}

		if (it == end) {
			// Begin new set
			TVector<T> v(1, t);
			set.insert(ymake_pair(t, ymake_pair(m_maxidx++, v)));
			m_inv[t] = t;
		}
	}
};

// Mainly for debugging
template<class T, class Eq>
yostream& operator << (yostream& stream, const Partition<T, Eq>& partition)
{
	stream << "Partition {\n";
	for (auto&& partitionElement : partition) {
		stream << "    Class " << partitionElement.second.first << " \"" << partitionElement.first << "\" { ";
		bool first = false;
		for (auto&& element : partitionElement.second.second) {
			if (first)
				stream << ", ";
			else
				first = true;
			stream << element;
		}
		stream << " }\n";
	}
	stream << "}";
	return stream;
}

}


#endif
