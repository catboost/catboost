/*
 * capture.h -- definition of CapturingScanner
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


#ifndef PIRE_EXTRA_CAPTURE_H
#define PIRE_EXTRA_CAPTURE_H


#include <contrib/libs/pire/pire/approx_matching.h>
#include <contrib/libs/pire/pire/scanners/loaded.h>
#include <contrib/libs/pire/pire/scanners/multi.h>
#include <contrib/libs/pire/pire/scanners/slow.h>
#include <contrib/libs/pire/pire/fsm.h>
#include <contrib/libs/pire/pire/re_lexer.h>
#include <contrib/libs/pire/pire/run.h>

#include <array>

namespace Pire {

/**
* A capturing scanner.
* Requires source FSM to be deterministic, matches input string
* against a single regexp (taking O(strlen(str)) time) and
* captures a substring between a single pair of parentheses.
*
* Requires regexp pattern to satisfy certain conditions
* (I still do not know exactly what they are :) )
*/
class CapturingScanner: public LoadedScanner {
public:
	enum {
		NoAction = 0,
		BeginCapture = 1,
		EndCapture   = 2,
		
		FinalFlag = 1
	};

	class State {
	public:
		bool Captured() const { return (m_begin != npos) && (m_end != npos); }
		size_t Begin() const { return m_begin; }
		size_t End() const { return m_end; }
	private:
		static const size_t npos = static_cast<size_t>(-1);
		size_t m_state;
		size_t m_begin;
		size_t m_end;
		size_t m_counter;
		friend class CapturingScanner;

#ifdef PIRE_DEBUG
		friend yostream& operator << (yostream& s, const State& state)
		{
			s << state.m_state;
			if (state.m_begin != State::npos || state.m_end != npos) {
				s << " [";
				if (state.m_begin != State::npos)
					s << 'b';
				if (state.m_end != State::npos)
					s << 'e';
				s << "]";
			}
			return s;
		}
#endif
	};

	void Initialize(State& state) const
	{
		state.m_state = m.initial;
		state.m_begin = state.m_end = State::npos;
		state.m_counter = 0;
	}

	void TakeAction(State& s, Action a) const
	{
		if ((a & BeginCapture) && !s.Captured())
			s.m_begin = s.m_counter - 1;
		else if (a & EndCapture) {
			if (s.m_end == State::npos)
				s.m_end = s.m_counter - 1;
			}
	}

	Char Translate(Char ch) const
	{
		return m_letters[static_cast<size_t>(ch)];
	}

	Action NextTranslated(State& s, unsigned char c) const
	{
		Transition x = reinterpret_cast<const Transition*>(s.m_state)[c];
		s.m_state += SignExtend(x.shift);
		++s.m_counter;

		return x.action;
	}

	Action Next(State& s, Char c) const
	{
		return NextTranslated(s, Translate(c));
	}

	Action Next(const State& current, State& n, Char c) const
	{
		n = current;
		return Next(n, c);
	}

	bool CanStop(const State& s) const
	{
		return Final(s);
	}

	bool Final(const State& s) const { return m_tags[(reinterpret_cast<Transition*>(s.m_state) - m_jumps) / m.lettersCount] & FinalFlag; }

	bool Dead(const State&) const { return false; }

	CapturingScanner() {}
	CapturingScanner(const CapturingScanner& s): LoadedScanner(s) {}
	explicit CapturingScanner(Fsm& fsm, size_t distance = 0)
	{
		if (distance) {
			fsm = CreateApproxFsm(fsm, distance);
		}
		fsm.Canonize();
		Init(fsm.Size(), fsm.Letters(), fsm.Initial());
		BuildScanner(fsm, *this);
	}
	
	void Swap(CapturingScanner& s) { LoadedScanner::Swap(s); }
	CapturingScanner& operator = (const CapturingScanner& s) { CapturingScanner(s).Swap(*this); return *this; }

	size_t StateIndex(const State& s) const { return StateIdx(s.m_state); }

private:

	friend void BuildScanner<CapturingScanner>(const Fsm&, CapturingScanner&);
};

enum RepetitionTypes { // They are sorted by their priorities
	NonGreedyRepetition,
	NoRepetition,
	GreedyRepetition,
};

/**
* A Slowcapturing scanner.
* Does not require FSM to be deterministic, uses O(|text| * |regexp|) time, but always returns the correct capture and supports both greedy and non-greedy quantifiers.
*/
class SlowCapturingScanner :  public SlowScanner {
public:
	enum {
		Nothing = 0,
		BeginCapture = 1,
		EndCapture   = 2,
		EndRepetition = 4,
		EndNonGreedyRepetition = 8,

		FinalFlag = 1,
	};

	const ui32 ActionsCapture = BeginCapture | EndCapture;

	class SingleState {
	public:
		bool Captured() const
		{
			return (m_begin != m_npos && m_end != m_npos);
		}

		bool HasBegin() const
		{
			return (m_begin != m_npos);
		}

		bool HasEnd() const
		{
			return (m_end != m_npos);
		}

		SingleState(size_t num = 0)
		{
			m_state = num;
			m_begin = m_npos;
			m_end = m_npos;
		}

		void SetBegin(size_t pos)
		{
			if (m_begin == m_npos)
				m_begin = pos;
		}

		void SetEnd(size_t pos)
		{
			if (m_end == m_npos)
				m_end = pos;
		}

		size_t Begin() const
		{
			return GetBegin();
		}

		size_t End() const
		{
			return GetEnd();
		}

		size_t GetBegin() const
		{
			return m_begin;
		}

		size_t GetEnd() const
		{
			return m_end;
		}

		size_t GetNum() const
		{
			return m_state;
		}

	private:
		size_t m_state;
		size_t m_begin;
		size_t m_end;
		static const size_t m_npos = static_cast<size_t>(-1);
	};

	class State {
	public:
		State()
			: m_strpos(0)
			, m_matched(false) {}

		size_t GetPos() const
		{
			return m_strpos;
		}

		const SingleState& GetState(size_t pos) const
		{
			return m_states[pos];
		}

		void SetPos(size_t newPos)
		{
			m_strpos = newPos;
		}

		void PushState(SingleState& st)
		{
			m_states.push_back(st);
		}

		void PushState(const SingleState& st)
		{
			m_states.push_back(st);
		}

		size_t GetSize() const
		{
			return m_states.size();
		}

		const TVector<SingleState>& GetStates() const
		{
			return m_states;
		}

		bool IsMatched() const
		{
			return m_matched;
		}

		const SingleState& GetMatched() const
		{
			return m_match;
		}

		void AddMatch(const SingleState& Matched)
		{
			m_matched = true;
			m_match = Matched;
		}

	private:
		TVector<SingleState> m_states;
		size_t m_strpos;
		bool m_matched;
		SingleState m_match;
	};

	class Transition {
	private:
		unsigned long m_stateto;
		Action m_action;

	public:
		unsigned long GetState() const
		{
			return m_stateto;
		}

		Action GetAction() const
		{
			return m_action;
		}

		Transition(unsigned long state, Action act = 0)
				: m_stateto(state)
				, m_action(act)
		{
		}
	};

	class PriorityStates {
	private:
		TVector<SingleState> m_nonGreedy;
		TVector<SingleState> m_nothing;
		TVector<SingleState> m_greedy;

	public:
		void Push(const SingleState& st, RepetitionTypes repetition)
		{
			Get(repetition).push_back(st);
		}

		TVector<SingleState>& Get(RepetitionTypes repetition)
		{
			switch (repetition) {
				case NonGreedyRepetition:
					return m_nonGreedy;
				case NoRepetition:
					return m_nothing;
				case GreedyRepetition:
					return m_greedy;
			}
		}

		const TVector<SingleState>& Get(RepetitionTypes repetition) const
		{
			switch (repetition) {
				case NonGreedyRepetition:
					return m_nonGreedy;
				case NoRepetition:
					return m_nothing;
				case GreedyRepetition:
					return m_greedy;
			}
		}
	};

	SlowScanner::State GetNextStates(const SingleState& cur, Char letter) const
	{
		SlowScanner::State st(GetSize());
		st.states.push_back(cur.GetNum());
		SlowScanner::State nextState(GetSize());
		SlowScanner::NextTranslated(st, nextState, letter);
		return nextState;
	}

	size_t GetPosition(const SingleState& state, Char letter) const
	{
		return state.GetNum() * GetLettersCount() + letter;
	}

	void NextStates(const SingleState& state, Char letter, TVector<Transition>& nextStates) const
	{
		if (IsMmaped()) {
			const size_t* pos = GetJumpPos() + GetPosition(state, letter);
			size_t posBegin = pos[0];
			size_t posEnd = pos[1];
			for (size_t i = posBegin; i < posEnd; ++i)
				nextStates.emplace_back(GetJump(i), GetAction(i));
		} else {
			size_t num = GetPosition(state, letter);
			const auto& jumpVec = GetJumpsVec(num);
			const auto& actionVec = GetActionsVec(num);
			for (size_t i = 0; i < jumpVec.size(); ++i)
				nextStates.emplace_back(jumpVec[i], actionVec[i]);
		}
	}

	void InsertStates(const PriorityStates& states, TVector<SingleState>& nonGreedy, TVector<SingleState>& nothing, TVector<SingleState>& greedy) const
	{
		for (auto& greed : {ymake_pair(&nonGreedy, NonGreedyRepetition), ymake_pair(&nothing, NoRepetition), ymake_pair(&greedy, GreedyRepetition)}) {
			auto& vec = greed.first;
			auto& tag = greed.second;
			vec->insert(vec->end(), states.Get(tag).begin(), states.Get(tag).end());
		}
	}

	void NextAndGetToGroups(PriorityStates& states, const SingleState& cur,
							Char letter, size_t pos, TVector<bool>& used) const
	{
		TVector<Transition> nextStates;
		NextStates(cur, letter, nextStates);
		for (const auto& trans : nextStates) {
			size_t st = trans.GetState();
			if (used[st])
				continue;
			used[st] = true;
			SingleState state(st);
			const auto& action = trans.GetAction();
			state.SetBegin(cur.GetBegin());
			state.SetEnd(cur.GetEnd());
			if (action & BeginCapture && !cur.HasBegin()) {
				state.SetBegin(pos);
			}
			if (action & EndCapture && !cur.HasEnd()) {
				state.SetEnd(pos);
			}
			PriorityStates statesInside;
			NextAndGetToGroups(statesInside, state, Translate(Epsilon), pos, used);
			statesInside.Push(state, NoRepetition);
			if (action & EndNonGreedyRepetition) {
				auto& nongreedy = states.Get(NonGreedyRepetition);
				InsertStates(statesInside, nongreedy, nongreedy, nongreedy);
			}
			else if (!(action & EndRepetition))
				InsertStates(statesInside, states.Get(NonGreedyRepetition), states.Get(NoRepetition), states.Get(GreedyRepetition));
			else {
				auto& greedy = states.Get(GreedyRepetition);
				InsertStates(statesInside, greedy, greedy, greedy);
			}
		}
	}

	bool Captured(const SingleState& st, bool& matched) const
	{
		matched = false;
		if (IsFinal(st.GetNum())) {
			matched = true;
			if (st.HasBegin())
				return true;
		}
		TVector<Transition> nextStates;
		NextStates(st, Translate(EndMark), nextStates);
		for (const auto& trans : nextStates)
		{
			size_t state = trans.GetState();
			if (IsFinal(state)) {
				matched = true;
				if (st.HasBegin() || (trans.GetAction() & ActionsCapture))
					return true;
			} else { // After EndMark there can be Epsilon-transitions to the Final State
				TVector<Transition> epsilonTrans;
				SingleState newSt(state);
				NextStates(newSt, Translate(Epsilon), epsilonTrans);
				for (auto new_trans : epsilonTrans) {
					size_t fin = new_trans.GetState();
					if (IsFinal(fin)) {
						matched = true;
						if (st.HasBegin() || (trans.GetAction() & ActionsCapture))
							return true;
					}
				}
			}
		}
		return false;
	}

	bool Matched(const SingleState& st) const
	{
		bool matched;
		Captured(st, matched);
		return matched;
	}

	bool GetCapture(const State& st, SingleState& final) const
	{
		size_t pos = 0;
		bool matched = false;
		bool ans = false;
		while (pos < st.GetSize() && !matched) {
			ans = Captured(st.GetState(pos), matched);
			++pos;
		}
		if (matched) {
			final = st.GetState(pos - 1);
			return ans;
		} else {
			if (st.IsMatched()) {
				final = st.GetMatched();
				return true;
			}
			return false;
		}
	}

	bool PushState(State& nlist, const SingleState& state) const
	{
		nlist.PushState(state);
		if (Matched(state)) {
			nlist.AddMatch(state);
			return true;
		}
		return false;
	}

	void UpdateNList(State& nlist, const PriorityStates& states) const
	{
		static constexpr std::array<RepetitionTypes, 3> m_type_by_priority{{NonGreedyRepetition, NoRepetition, GreedyRepetition}};
		for (const auto type : m_type_by_priority) {
			for (const auto& state : states.Get(type)) {
				if (PushState(nlist, state)) // Because we have strict priorities, after matching some state, we can be sure, that not states after will be better
					return;
			}
		}
	}

	void Initialize(State& nlist) const
	{
		PriorityStates states;
		SingleState init(GetStart());
		TVector<bool> used(GetSize());
		NextAndGetToGroups(states, init, Translate(BeginMark), 0, used);
		NextAndGetToGroups(states, 0, Translate(BeginMark), 0, used);
		UpdateNList(nlist, states);
	}

	Action NextTranslated(State& clist, Char letter) const
	{
		State nlist;
		if (clist.IsMatched())
			nlist.AddMatch(clist.GetMatched());
		nlist.SetPos(clist.GetPos() + 1);
		size_t strpos = nlist.GetPos();
		TVector<bool> used(GetSize());
		size_t pos = 0;
		while (pos < clist.GetSize()) {
			PriorityStates states;
			NextAndGetToGroups(states, clist.GetState(pos), letter, strpos, used);
			UpdateNList(nlist, states);
			++pos;
		}
		DoSwap(clist, nlist);
		return 0;
	}

	void TakeAction(State&, Action) const {}

	Action Next(State& st, Char letter) const
	{
		return NextTranslated(st, Translate(letter));
	}

public:
	SlowCapturingScanner()
		: SlowScanner(true)
	{
	}

	SlowCapturingScanner(Fsm& fsm, size_t distance = 0)
		: SlowScanner(fsm, true, false, distance)
	{
	}
};

namespace Features {
	Feature::Ptr Capture(size_t pos);
}

}


#endif
