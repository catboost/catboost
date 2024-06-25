/*
 * glue.h -- scanner agglutination task, which can be used as
 *           a parameter to Determine().
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


#ifndef PIRE_GLUE_H
#define PIRE_GLUE_H


#include <contrib/libs/pire/pire/stub/stl.h>
#include "partition.h"

namespace Pire {
namespace Impl {

template <class Scanner>
class LettersEquality: public ybinary_function<Char, Char, bool> {
public:
    LettersEquality(typename Scanner::Letter* lhs, typename Scanner::Letter* rhs): m_lhs(lhs), m_rhs(rhs) {}

    bool operator()(Char a, Char b) const
    {
        return m_lhs[a] == m_lhs[b] && m_rhs[a] == m_rhs[b];
    }

private:
    typename Scanner::Letter* m_lhs;
    typename Scanner::Letter* m_rhs;
};	

// This lookup table is used instead of std::map.
// The key idea is to specify size which is a power of 2 in order to use >> and | instead of
// divisions and remainders.
// NB: it mimics limited std::map<> behaviour, hence stl-like method names and typedefs.
template <size_t N, class State>
class GluedStateLookupTable {
public:
	static const size_t MaxSize = N;
	typedef ypair<State, State> key_type;
	typedef size_t mapped_type;
	typedef ypair<key_type, mapped_type> value_type;
	typedef value_type* iterator;
	typedef const value_type* const_iterator;

	GluedStateLookupTable()
		: mMap(new value_type[N])
		, mFilled(N, false)
	{}

	~GluedStateLookupTable() = default;
	
	const_iterator end() const {
		return mMap.Get() + MaxSize;
	}
	// Note that in fact mMap is sparsed and traditional [begin,end)
	// traversal is unavailable; hence no begin() method here.
	// end() is only valid for comparing with find() result.
	const_iterator find(const key_type& st) const {
		size_t ind = Search(st);
		return mFilled[ind] ? (mMap.Get() + ind) : end();
	}

	ypair<iterator, bool> insert(const value_type& v) {
		size_t ind = Search(v.first);
		if (!mFilled[ind]) {
			mMap[ind] = v;
			mFilled[ind] = true;
			return ymake_pair(mMap.Get() + ind, true);
		} else
			return ymake_pair(mMap.Get() + ind, false);
	}

private:
	size_t Search(const key_type& st) const {
		size_t startInd = (Hash(st) % N);
		for (size_t ind = startInd; ind != (startInd + N - 1) % N; ind = (ind + 1) % N) {
			if (!mFilled[ind] || mMap[ind].first == st) {
				return ind;
			}
		}
		return (size_t)-1;
	}

	static size_t Hash(const key_type& st) {
		return size_t((st.first >> 2) ^ (st.second >> 4) ^ (st.second << 10));
	}

	TArrayHolder<value_type> mMap;
	TVector<bool> mFilled;

	// Noncopyable
	GluedStateLookupTable(const GluedStateLookupTable&);
	GluedStateLookupTable& operator = (const GluedStateLookupTable&);
};

template<class Scanner>
class ScannerGlueCommon {
public:
	typedef Partition< Char, Impl::LettersEquality<Scanner> > LettersTbl;

	typedef ypair<typename Scanner::InternalState, typename Scanner::InternalState> State;
	ScannerGlueCommon(const Scanner& lhs, const Scanner& rhs, const LettersTbl& letters)
		: m_lhs(lhs)
		, m_rhs(rhs)
		, m_letters(letters)
	{
		// Form a new letters partition
		for (unsigned ch = 0; ch < MaxChar; ++ch)
			if (ch != Epsilon)
				m_letters.Append(ch);
	}

	const LettersTbl& Letters() const { return m_letters; }

	const Scanner& Lhs() const { return m_lhs; }
	const Scanner& Rhs() const { return m_rhs; }

	State Initial() const { return State(Lhs().m.initial, Rhs().m.initial); }

	State Next(State state, Char letter) const
	{
		Lhs().Next(state.first, letter);
		Rhs().Next(state.second, letter);
		return state;
	}

	bool IsRequired(const State& /*state*/) const { return true; }

	typedef Scanner Result;
	const Scanner& Success() const { return *m_result; }
	Scanner Failure() const { return Scanner(); }

protected:
	Scanner& Sc() { return *m_result; }
	void SetSc(THolder<Scanner>&& sc) { m_result = std::move(sc); }

private:
	const Scanner& m_lhs;
	const Scanner& m_rhs;
	LettersTbl m_letters;
	THolder<Scanner> m_result;
};

}	
}

#endif
