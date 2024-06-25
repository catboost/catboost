/*
 * slow.h -- definition of the SlowScanner
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


#ifndef PIRE_SCANNERS_SLOW_H
#define PIRE_SCANNERS_SLOW_H

#include <contrib/libs/pire/pire/approx_matching.h>
#include <contrib/libs/pire/pire/partition.h>
#include <contrib/libs/pire/pire/vbitset.h>
#include <contrib/libs/pire/pire/fsm.h>
#include <contrib/libs/pire/pire/run.h>
#include <contrib/libs/pire/pire/stub/saveload.h>
#include <contrib/libs/pire/pire/stub/stl.h>

#include "common.h"

#ifdef PIRE_DEBUG
#include <iostream>
#include <contrib/libs/pire/pire/stub/lexical_cast.h>
#endif

namespace Pire {

/**
 * A 'slow' scanner.
 * Takes O( str.length() * this->m_states.size() ) time to scan string,
 * but does not require FSM to be deterministic.
 * Thus can be used to handle something sorta /x.{40}$/,
 * where deterministic FSM contains 2^40 states and hence cannot fit
 * in memory.
 */
class SlowScanner {
public:
	typedef size_t      Transition;
	typedef ui16        Letter;
	typedef ui32        Action;
	typedef ui8         Tag;

	enum {
		FinalFlag = 1,
		DeadFlag  = 0
	};

	struct State {
		TVector<unsigned> states;
		BitSet flags;

		State() {}
		State(size_t size): flags(size) { states.reserve(size); }
		void Swap(State& s) { states.swap(s.states); flags.Swap(s.flags); }

#ifdef PIRE_DEBUG
		friend yostream& operator << (yostream& stream, const State& state) { return stream << Join(state.states.begin(), state.states.end(), ", "); }
#endif
	};

	SlowScanner(bool needActions = false) {
		Alias(Null());
		need_actions = needActions;
	}

	size_t GetLettersCount() const {return m.lettersCount; };

	size_t Size() const { return GetSize(); }
	size_t GetSize() const { return m.statesCount; }
	bool Empty() const { return m_finals == Null().m_finals; }

	size_t Id() const {return (size_t) -1;}
	size_t RegexpsCount() const { return Empty() ? 0 : 1; }

	void Initialize(State& state) const
	{
		state.states.clear();
		state.states.reserve(m.statesCount);
		state.states.push_back(m.start);
		BitSet(m.statesCount).Swap(state.flags);
	}

	Char Translate(Char ch) const
	{
		return m_letters[static_cast<size_t>(ch)];
	}

	Action NextTranslated(const State& current, State& next, Char l) const
	{
		next.flags.Clear();
		next.states.clear();
		for (auto&& state : current.states) {
			const unsigned* begin = 0;
			const unsigned* end = 0;
			if (!m_vecptr) {
				const size_t* pos = m_jumpPos + state * m.lettersCount + l;
				begin = m_jumps + pos[0];
				end = m_jumps + pos[1];
			} else {
				const auto& v = (*m_vecptr)[state * m.lettersCount + l];
				if (!v.empty()) {
					begin = &v[0];
					end = &v[0] + v.size();
				}
			}

			for (; begin != end; ++begin)
				if (!next.flags.Test(*begin)) {
					next.flags.Set(*begin);
					next.states.push_back(*begin);
				}
		}

		return 0;
	}

	Action Next(const State& current, State& next, Char c) const
	{
		return NextTranslated(current, next, Translate(c));
	}

	bool TakeAction(State&, Action) const { return false; }

	Action NextTranslated(State& s, Char l) const
	{
		State dest(m.statesCount);
		Action a = NextTranslated(s, dest, l);
		s.Swap(dest);
		return a;
	}

	Action Next(State& s, Char c) const
	{
		return NextTranslated(s, Translate(c));
	}

	bool Final(const State& s) const
	{
		for (auto&& state : s.states)
			if (m_finals[state])
				return true;
		return false;
	}

	bool Dead(const State&) const
	{
		return false;
	}

	ypair<const size_t*, const size_t*> AcceptedRegexps(const State& s) const {
		return Final(s) ? Accept() : Deny();
	}

	bool CanStop(const State& s) const {
		return Final(s);
	}

	const void* Mmap(const void* ptr, size_t size)
	{
		Impl::CheckAlign(ptr);
		SlowScanner s;
		const size_t* p = reinterpret_cast<const size_t*>(ptr);

		Impl::ValidateHeader(p, size, ScannerIOTypes::SlowScanner, sizeof(s.m));
		Locals* locals;
		Impl::MapPtr(locals, 1, p, size);
		memcpy(&s.m, locals, sizeof(s.m));

		bool empty = *((const bool*) p);
		Impl::AdvancePtr(p, size, sizeof(empty));
		Impl::AlignPtr(p, size);

		if (empty)
			s.Alias(Null());
		else {
			s.m_vecptr = 0;
			Impl::MapPtr(s.m_letters, MaxChar, p, size);
			Impl::MapPtr(s.m_finals, s.m.statesCount, p, size);
			Impl::MapPtr(s.m_jumpPos, s.m.statesCount * s.m.lettersCount + 1, p, size);
			Impl::MapPtr(s.m_jumps, s.m_jumpPos[s.m.statesCount * s.m.lettersCount], p, size);
			if (need_actions)
				Impl::MapPtr(s.m_actions, s.m_jumpPos[s.m.statesCount * s.m.lettersCount], p, size);
			Swap(s);
		}
		return (const void*) p;
	}

	void Swap(SlowScanner& s)
	{
		DoSwap(m_finals, s.m_finals);
		DoSwap(m_jumps, s.m_jumps);
		DoSwap(m_actions, s.m_actions);
		DoSwap(m_jumpPos, s.m_jumpPos);
		DoSwap(m.statesCount, s.m.statesCount);
		DoSwap(m.lettersCount, s.m.lettersCount);
		DoSwap(m.start, s.m.start);
		DoSwap(m_letters, s.m_letters);
		DoSwap(m_pool, s.m_pool);
		DoSwap(m_vec, s.m_vec);

		DoSwap(m_vecptr, s.m_vecptr);
		DoSwap(need_actions, s.need_actions);
		DoSwap(m_actionsvec, s.m_actionsvec);
		if (m_vecptr == &s.m_vec)
			m_vecptr = &m_vec;
		if (s.m_vecptr == &m_vec)
			s.m_vecptr = &s.m_vec;
	}

	SlowScanner(const SlowScanner& s)
		: m(s.m)
		, m_vec(s.m_vec)
		, need_actions(s.need_actions)
		, m_actionsvec(s.m_actionsvec)
	{
		if (s.m_vec.empty()) {
			// Empty or mmap()-ed scanner, just copy pointers
			m_finals = s.m_finals;
			m_jumps = s.m_jumps;
			m_actions = s.m_actions;
			m_jumpPos = s.m_jumpPos;
			m_letters = s.m_letters;
			m_vecptr = 0;
		} else {
			// In-memory scanner, perform deep copy
			alloc(m_letters, MaxChar);
			memcpy(m_letters, s.m_letters, sizeof(*m_letters) * MaxChar);
			m_jumps = 0;
			m_jumpPos = 0;
			m_actions = 0;
			alloc(m_finals, m.statesCount);
			memcpy(m_finals, s.m_finals, sizeof(*m_finals) * m.statesCount);
			m_vecptr = &m_vec;
		}
	}

	explicit SlowScanner(Fsm& fsm, bool needActions = false, bool removeEpsilons = true, size_t distance = 0)
		: need_actions(needActions)
	{
		if (distance) {
			fsm = CreateApproxFsm(fsm, distance);
		}
		if (removeEpsilons)
			fsm.RemoveEpsilons();
		fsm.Sparse(!removeEpsilons);

		m.statesCount = fsm.Size();
		m.lettersCount = fsm.Letters().Size();

		m_vec.resize(m.statesCount * m.lettersCount);
		if (need_actions)
			m_actionsvec.resize(m.statesCount * m.lettersCount);
		m_vecptr = &m_vec;
		alloc(m_letters, MaxChar);
		m_jumps = 0;
		m_actions = 0;
		m_jumpPos = 0;
		alloc(m_finals, m.statesCount);

		// Build letter translation table
		Fill(m_letters, m_letters + MaxChar, 0);
		for (auto&& letter : fsm.Letters())
			for (auto&& character : letter.second.second)
				m_letters[character] = letter.second.first;

		m.start = fsm.Initial();
		BuildScanner(fsm, *this);
	}


	SlowScanner& operator = (const SlowScanner& s) { SlowScanner(s).Swap(*this); return *this; }

	~SlowScanner()
	{
		for (auto&& i : m_pool)
			free(i);
	}

	void Save(yostream*) const;
	void Load(yistream*);

	const State& StateIndex(const State& s) const { return s; }

protected:
	bool IsMmaped() const
	{
		return (!m_vecptr);
	}

	size_t GetJump(size_t pos) const
	{
		return m_jumps[pos];
	}

	Action& GetAction(size_t pos) const
	{
		return m_actions[pos];
	}

	const TVector<Action>& GetActionsVec(size_t from) const
	{
		return m_actionsvec[from];
	}

	const TVector<unsigned int>& GetJumpsVec(size_t from) const
	{
		return m_vec[from];
	}

	size_t* GetJumpPos() const
	{
		return m_jumpPos;
	}

	size_t GetStart() const
	{
		return m.start;
	}

	bool IsFinal(size_t pos) const
	{
		return m_finals[pos];
	}

private:

	struct Locals {
		size_t statesCount;
		size_t lettersCount;
		size_t start;
	} m;

	bool* m_finals;
	unsigned* m_jumps;
	Action* m_actions;
	size_t* m_jumpPos;
	size_t* m_letters;

	TVector<void*> m_pool;
	TVector< TVector<unsigned> > m_vec, *m_vecptr;

	bool need_actions;
	TVector<TVector<Action>> m_actionsvec;
	static const SlowScanner& Null();

	template<class T> void alloc(T*& p, size_t size)
	{
		p = static_cast<T*>(malloc(size * sizeof(T)));
		memset(p, 0, size * sizeof(T));
		m_pool.push_back(p);
	}

	void Alias(const SlowScanner& s)
	{
		memcpy(&m, &s.m, sizeof(m));
		m_vec.clear();
		need_actions = s.need_actions;
		m_actionsvec.clear();
		m_finals = s.m_finals;
		m_jumps = s.m_jumps;
		m_actions = s.m_actions;
		m_jumpPos = s.m_jumpPos;
		m_letters = s.m_letters;
		m_vecptr = s.m_vecptr;
		m_pool.clear();
	}

	void SetJump(size_t oldState, Char c, size_t newState, unsigned long action)
	{
		Y_ASSERT(!m_vec.empty());
		Y_ASSERT(oldState < m.statesCount);
		Y_ASSERT(newState < m.statesCount);

		size_t idx = oldState * m.lettersCount + m_letters[c];
		m_vec[idx].push_back(newState);
		if (need_actions)
			m_actionsvec[idx].push_back(action);
	}

	unsigned long RemapAction(unsigned long action) { return action; }

	void SetInitial(size_t state) { m.start = state; }
	void SetTag(size_t state, ui8 tag) { m_finals[state] = (tag != 0); }

	void FinishBuild() {}

	static ypair<const size_t*, const size_t*> Accept()
	{
		static size_t v[1] = { 0 };

		return ymake_pair(v, v + 1);
	}

	static ypair<const size_t*, const size_t*> Deny()
	{
		static size_t v[1] = { 0 };
		return ymake_pair(v, v);
	}

	friend void BuildScanner<SlowScanner>(const Fsm&, SlowScanner&);
};

template<>
inline SlowScanner Fsm::Compile(size_t distance) {
	return SlowScanner(*this, false, true, distance);
}

inline const SlowScanner& SlowScanner::Null()
{
	static const SlowScanner n = Fsm::MakeFalse().Compile<SlowScanner>();
	return n;
}

#ifndef PIRE_DEBUG
/// A specialization of Run(), since its state is much heavier than other ones
/// and we thus want to avoid copying states.
template<>
inline void Run<SlowScanner>(const SlowScanner& scanner, SlowScanner::State& state, TStringBuf str)
{
	SlowScanner::State temp;
	scanner.Initialize(temp);

	SlowScanner::State* src = &state;
	SlowScanner::State* dest = &temp;

	for (auto it = str.begin(); it != str.end(); ++it) {
		scanner.Next(*src, *dest, static_cast<unsigned char>(*it));
		DoSwap(src, dest);
	}
	if (src != &state)
		state = *src;
}
#endif

}


#endif
