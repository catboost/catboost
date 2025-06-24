/*
 * pair.h -- definition of the pair of scanners
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

#ifndef PIRE_SCANNER_PAIR_INCLUDED
#define PIRE_SCANNER_PAIR_INCLUDED

namespace Pire {

        /**
         * A pair of scanner, providing the interface of a scanner itself.
         * If you need to run two scanners on the same string, using ScannerPair
         * is usually faster then running those scanners sequentially.
         */
	template<class Scanner1, class Scanner2>
	class ScannerPair {
	public:
		typedef ypair<typename Scanner1::State, typename Scanner2::State> State;
		typedef ypair<typename Scanner1::Action, typename Scanner2::Action> Action;

		ScannerPair()
			: m_scanner1()
			, m_scanner2()
		{
		}
		ScannerPair(const Scanner1& s1, const Scanner2& s2)
			: m_scanner1(&s1)
			, m_scanner2(&s2)
		{
		}

		void Initialize(State& state) const
		{
			m_scanner1->Initialize(state.first);
			m_scanner2->Initialize(state.second);
		}

		Action Next(State& state, Char ch) const
		{
			return ymake_pair(
				m_scanner1->Next(state.first, ch),
				m_scanner2->Next(state.second, ch)
			);
		}

		void TakeAction(State& s, Action a) const
		{
			m_scanner1->TakeAction(s.first, a.first);
			m_scanner2->TakeAction(s.second, a.second);
		}

		bool Final(const State& state) const
		{
			return m_scanner1->Final(state.first) || m_scanner2->Final(state.second);
		}

		bool Dead(const State& state) const
		{
			return m_scanner1->Dead(state.first) && m_scanner2->Dead(state.second);
		}

		ypair<size_t, size_t> StateIndex(const State& state) const
		{
			return ymake_pair(m_scanner1->StateIndex(state.first), m_scanner2->StateIndex(state.second));
		}

		Scanner1& First() { return *m_scanner1; }
		Scanner2& Second() { return *m_scanner2; }

		const Scanner1& First() const { return *m_scanner1; }
		const Scanner2& Second() const { return *m_scanner2; }

	private:
		const Scanner1* m_scanner1;
		const Scanner2* m_scanner2;
	};

        
}

#endif
