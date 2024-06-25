/*
 * count.h -- definition of the counting scanner
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


#ifndef PIRE_EXTRA_COUNT_H
#define PIRE_EXTRA_COUNT_H

#include <contrib/libs/pire/pire/scanners/loaded.h>
#include <contrib/libs/pire/pire/fsm.h>

#include <algorithm>

namespace Pire {
class Fsm;

namespace Impl {
	template<class T>
	class ScannerGlueCommon;

	template<class T>
	class CountingScannerGlueTask;

	class NoGlueLimitCountingScannerGlueTask;

	template <class AdvancedScanner>
	AdvancedScanner MakeAdvancedCountingScanner(const Fsm& re, const Fsm& sep, bool* simple);
};

template<size_t I>
class IncrementPerformer {
public:
	template<typename State, typename Action>
	PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	static void Do(State& s, Action mask)
	{
		if (mask & (1 << (I - 1))) {
			Increment(s);
		}
		IncrementPerformer<I - 1>::Do(s, mask);
	}

private:
	template<typename State>
	PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	static void Increment(State& s)
	{
		++s.m_current[I - 1];
	}
};

template<>
class IncrementPerformer<0> {
public:
	template<typename State, typename Action>
	PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	static void Do(State&, Action)
	{
	}
};

template<size_t I>
class ResetPerformer {
public:
	template<typename State, typename Action>
	PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	static void Do(State& s, Action mask)
	{
		if (mask & (1 << (LoadedScanner::MAX_RE_COUNT + (I - 1))) && s.m_current[I - 1]) {
			Reset(s);
		}
		ResetPerformer<I - 1>::Do(s, mask);
	}

private:
	template<typename State>
	PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	static void Reset(State& s)
	{
		s.m_total[I - 1] = ymax(s.m_total[I - 1], s.m_current[I - 1]);
		s.m_current[I - 1] = 0;
	}
};

template<>
class ResetPerformer<0> {
public:
	template<typename State, typename Action>
	PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	static void Do(State&, Action)
	{
	}
};

/**
 * A scanner which counts occurences of the
 * given regexp separated by another regexp
 * in input text.
 */
template<class DerivedScanner, class State>
class BaseCountingScanner: public LoadedScanner {
public:
	enum {
		IncrementAction = 1,
		ResetAction = 2,

		FinalFlag = 0,
		DeadFlag = 1,
	};

	void Initialize(State& state) const
	{
		state.m_state = m.initial;
		memset(&state.m_current, 0, sizeof(state.m_current));
		memset(&state.m_total, 0, sizeof(state.m_total));
		state.m_updatedMask = 0;
	}

	PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	void TakeAction(State& s, Action a) const
	{
		static_cast<const DerivedScanner*>(this)->template TakeActionImpl<MAX_RE_COUNT>(s, a);
	}

	bool CanStop(const State&) const { return false; }

	Char Translate(Char ch) const
	{
		return m_letters[static_cast<size_t>(ch)];
	}

	Action NextTranslated(State& s, Char c) const
	{
		Transition x = reinterpret_cast<const Transition*>(s.m_state)[c];
		s.m_state += SignExtend(x.shift);
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

	bool Final(const State& /*state*/) const { return false; }

	bool Dead(const State&) const { return false; }

	using LoadedScanner::Swap;

	size_t StateIndex(const State& s) const { return StateIdx(s.m_state); }

protected:
	using LoadedScanner::Init;
	using LoadedScanner::InternalState;

	template<size_t ActualReCount>
	void PerformIncrement(State& s, Action mask) const
	{
		if (mask) {
			IncrementPerformer<ActualReCount>::Do(s, mask);
			s.m_updatedMask |= ((size_t)mask) << MAX_RE_COUNT;
		}
	}

	template<size_t ActualReCount>
	void PerformReset(State& s, Action mask) const
	{
		mask &= s.m_updatedMask;
		if (mask) {
			ResetPerformer<ActualReCount>::Do(s, mask);
			s.m_updatedMask &= (Action)~mask;
		}
	}

	void Next(InternalState& s, Char c) const
	{
		Transition x = reinterpret_cast<const Transition*>(s)[Translate(c)];
		s += SignExtend(x.shift);
	}
};

template <size_t MAX_RE_COUNT>
class CountingState {
public:
	size_t Result(int i) const { return ymax(m_current[i], m_total[i]); }
private:
	using InternalState = LoadedScanner::InternalState;
	InternalState m_state;
	ui32 m_current[MAX_RE_COUNT];
	ui32 m_total[MAX_RE_COUNT];
	size_t m_updatedMask;

	template <class DerivedScanner, class State>
	friend class BaseCountingScanner;

	template<size_t I>
	friend class IncrementPerformer;

	template<size_t I>
	friend class ResetPerformer;

#ifdef PIRE_DEBUG
	friend yostream& operator << (yostream& s, const State& state)
		{
			s << state.m_state << " ( ";
			for (size_t i = 0; i < MAX_RE_COUNT; ++i)
				s << state.m_current[i] << '/' << state.m_total[i] << ' ';
			return s << ')';
		}
#endif
};


class CountingScanner : public BaseCountingScanner<CountingScanner, CountingState<LoadedScanner::MAX_RE_COUNT>> {
public:
	using State = CountingState<MAX_RE_COUNT>;
	enum {
		Matched = 2,
	};
	
	CountingScanner() {}
	CountingScanner(const Fsm& re, const Fsm& sep);

	static CountingScanner Glue(const CountingScanner& a, const CountingScanner& b, size_t maxSize = 0);

	template<size_t ActualReCount>
	PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	void TakeActionImpl(State& s, Action a) const
	{
		if (a & IncrementMask)
			PerformIncrement<ActualReCount>(s, a);
		if (a & ResetMask)
			PerformReset<ActualReCount>(s, a);
	}

private:
	Action RemapAction(Action action)
	{
		if (action == (Matched | DeadFlag))
			return 1;
		else if (action == DeadFlag)
			return 1 << MAX_RE_COUNT;
		else
			return 0;
	}

	friend void BuildScanner<CountingScanner>(const Fsm&, CountingScanner&);
	friend class Impl::ScannerGlueCommon<CountingScanner>;
	friend class Impl::CountingScannerGlueTask<CountingScanner>;
};

class AdvancedCountingScanner : public BaseCountingScanner<AdvancedCountingScanner, CountingState<LoadedScanner::MAX_RE_COUNT>> {
public:
	using State = CountingState<MAX_RE_COUNT>;

	AdvancedCountingScanner() {}
	AdvancedCountingScanner(const Fsm& re, const Fsm& sep, bool* simple = nullptr);

	static AdvancedCountingScanner Glue(const AdvancedCountingScanner& a, const AdvancedCountingScanner& b, size_t maxSize = 0);

	template<size_t ActualReCount>
	PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	void TakeActionImpl(State& s, Action a) const
	{
		if (a & ResetMask) {
			PerformReset<ActualReCount>(s, a);
		}
		if (a & IncrementMask) {
			PerformIncrement<ActualReCount>(s, a);
		}
	}

private:
	Action RemapAction(Action action)
	{
		Action result = 0;
		if (action & ResetAction) {
			result = 1 << MAX_RE_COUNT;
		}
		if (action & IncrementAction) {
			result |= 1;
		}
		return result;
	}

	friend class Impl::ScannerGlueCommon<AdvancedCountingScanner>;
	friend class Impl::CountingScannerGlueTask<AdvancedCountingScanner>;
	friend AdvancedCountingScanner Impl::MakeAdvancedCountingScanner<AdvancedCountingScanner>(const Fsm&, const Fsm&, bool*);
};

class NoGlueLimitCountingState {
public:
	size_t Result(int i) const { return ymax(m_current[i], m_total[i]); }
    void Initialize(size_t initial, size_t regexpsCount) {
		m_state = initial;
		m_current.assign(regexpsCount, 0);
		m_total.assign(regexpsCount, 0);
	}
	PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	void Reset(size_t regexpId) {
		m_current[regexpId] = 0;
	}
	PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	void Increment(size_t regexp_id) {
		++m_current[regexp_id];
		m_total[regexp_id] = ymax(m_total[regexp_id], m_current[regexp_id]);
	}

	template<size_t I>
	friend class IncrementPerformer;

	template<size_t I>
	friend class ResetPerformer;

private:
	LoadedScanner::InternalState m_state;
	TVector<ui32> m_current;
	TVector<ui32> m_total;

	template <class DerivedScanner, class State>
	friend class BaseCountingScanner;

#ifdef PIRE_DEBUG
	yostream& operator << (yostream& s, const State& state)
		{
			s << state.m_state << " ( ";
			for (size_t i = 0; i < state.m_current.size(); ++i)
				s << state.m_current[i] << '/' << state.m_total[i] << ' ';
			return s << ')';
		}
#endif
};


class NoGlueLimitCountingScanner : public BaseCountingScanner<NoGlueLimitCountingScanner, NoGlueLimitCountingState> {
public:
	using State = NoGlueLimitCountingState;
	using ActionIndex = ui32;
	using TActionsBuffer = std::unique_ptr<ActionIndex[]>;

private:
	TActionsBuffer ActionsBuffer;
	const ActionIndex* Actions = nullptr;
	bool AdvancedScannerCompatibilityMode = false;

public:
	NoGlueLimitCountingScanner() = default;
	NoGlueLimitCountingScanner(const Fsm& re, const Fsm& sep, bool* simple = nullptr);
	NoGlueLimitCountingScanner(const NoGlueLimitCountingScanner& rhs)
	    : BaseCountingScanner(rhs)
	    , AdvancedScannerCompatibilityMode(rhs.AdvancedScannerCompatibilityMode)
	{
		if (rhs.ActionsBuffer) {
			Y_ASSERT(rhs.Actions);
			ActionsBuffer = TActionsBuffer(new ActionIndex [*rhs.Actions]);
			std::copy_n(rhs.ActionsBuffer.get(), *rhs.Actions, ActionsBuffer.get());
			Actions = ActionsBuffer.get();
		} else {
			Actions = rhs.Actions;
		}
	}

	NoGlueLimitCountingScanner(NoGlueLimitCountingScanner&& other) : BaseCountingScanner() {
		Swap(other);
	}

	NoGlueLimitCountingScanner& operator=(NoGlueLimitCountingScanner rhs) {
		Swap(rhs);
		return *this;
	}

	void Swap(NoGlueLimitCountingScanner& s) {
		LoadedScanner::Swap(s);
		DoSwap(ActionsBuffer, s.ActionsBuffer);
		DoSwap(Actions, s.Actions);
		DoSwap(AdvancedScannerCompatibilityMode, s.AdvancedScannerCompatibilityMode);
	}

	void Initialize(State& state) const
	{
		state.Initialize(m.initial, RegexpsCount());
	}

	template <size_t ActualReCount>
	PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	void TakeActionImpl(State& s, Action a) const
	{
		if (!a) {
			return;
		}
		if (AdvancedScannerCompatibilityMode) {
			AdvancedScannerTakeActionImpl<ActualReCount>(s, a);
			return;
		}
		// Note: it's important to perform resets before increments,
		// as it's possible for one repetition group to stop and another begin at the same symbol
		if (Actions) {
			auto action = Actions + a;
			for (auto reset_count = *action++; reset_count--;) {
				s.Reset(*action++);
			}
			for (auto inc_count = *action++; inc_count--;) {
				s.Increment(*action++);
			}
		} else {
			Y_ASSERT(RegexpsCount() == 1);
			if (a & ResetAction) {
				s.Reset(0);
			}
			if (a & IncrementAction) {
				s.Increment(0);
			}
		}
	}

	void Save(yostream* s) const;

	void Load(yistream* s);

	const void* Mmap(const void* ptr, size_t size);

	static NoGlueLimitCountingScanner Glue(const NoGlueLimitCountingScanner& a, const NoGlueLimitCountingScanner& b, size_t maxSize = 0);

private:
	Action RemapAction(Action action)
	{
		return action;
	}

	template <class Iterator>
	void GetActions(Action a, ActionIndex id_shift, Iterator output_resets, Iterator output_increments) const {
		if (!a) {
			return;
		}
		if (!Actions) {
			if (a & ResetAction) {
				*output_resets++ = id_shift;
			}
			if (a & NoGlueLimitCountingScanner::IncrementAction) {
				*output_increments++ = id_shift;
			}
			return;
		}
		auto action = Actions + a;
		for (auto output : {output_resets, output_increments}) {
			for (auto count = *action++; count--;) {
				*output++ = *action++ + id_shift;
			}
		}
	}

	void AcceptActions(const TVector<ActionIndex>& actions) {
		Y_ASSERT(!Actions);
		Y_ASSERT(!actions.empty());
		Y_ASSERT(actions[0] == actions.size());

		ActionsBuffer = TActionsBuffer(new ActionIndex[actions.size()]);
		std::copy(actions.begin(), actions.end(), ActionsBuffer.get());
		Actions = ActionsBuffer.get();
	}

	template <size_t ActualReCount>
	void AdvancedScannerTakeActionImpl(State& s, Action a) const {
		if (a & ResetMask) {
			ResetPerformer<ActualReCount>::Do(s, a);
		}
		if (a & IncrementMask) {
			IncrementPerformer<ActualReCount>::Do(s, a);
		}
	}

	friend class Impl::ScannerGlueCommon<NoGlueLimitCountingScanner>;
	friend class Impl::CountingScannerGlueTask<NoGlueLimitCountingScanner>;
	friend class Impl::NoGlueLimitCountingScannerGlueTask;
	friend NoGlueLimitCountingScanner Impl::MakeAdvancedCountingScanner<NoGlueLimitCountingScanner>(const Fsm&, const Fsm&, bool*);
};

}

#endif
