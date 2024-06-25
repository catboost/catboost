/*
 * count.cpp -- CountingScanner compiling routine
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


#include "count.h"

#include <contrib/libs/pire/pire/fsm.h>
#include <contrib/libs/pire/pire/determine.h>
#include <contrib/libs/pire/pire/glue.h>
#include <contrib/libs/pire/pire/minimize.h>
#include <contrib/libs/pire/pire/stub/lexical_cast.h>
#include <contrib/libs/pire/pire/stub/stl.h>

#include <tuple>

namespace Pire {

namespace Impl {

typedef LoadedScanner::Action Action;
typedef TMap<Char, Action> TransitionTagRow;
typedef TVector<TransitionTagRow> TransitionTagTable;

class CountingFsmTask;

class CountingFsm {
public:
	typedef Fsm::LettersTbl LettersTbl;

	enum {
		NotMatched = 1 << 0,
		Matched = 1 << 1,
		Separated = 1 << 2,
	};

	explicit CountingFsm(Fsm re, Fsm sep)
		: mFsm(std::move(re))
	{
		mFsm.Canonize();
		const auto reMatchedStates = mFsm.Finals();

		sep.Canonize();
		for (size_t state = 0; state < sep.Size(); ++state) {
			sep.SetTag(state, Separated);
		}
		mFsm += sep;

		mReInitial = mFsm.Initial();
		const auto allowEmptySeparator = sep.IsFinal(sep.Initial());
		for (auto reMatchedState : reMatchedStates) {
			mFsm.SetTag(reMatchedState, Matched);
			if (allowEmptySeparator) {
				mFsm.SetFinal(reMatchedState, true);
			}
		}

		mFsm.PrependAnything();
		mFsm.RemoveEpsilons();
	}

	const LettersTbl& Letters() const {
		return mFsm.Letters();
	}

	const Fsm& Determined() const {
		return mDetermined;
	}

	Action Output(size_t from, Char letter) const {
		const auto& row = mActions[from];
		const auto it = row.find(letter);
		if (it != row.end()) {
			return it->second;
		} else {
			return 0;
		}
	}

	bool Simple() const {
		return mSimple;
	}

	bool Determine();
	void Minimize();

private:
	void SwapTaskOutputs(CountingFsmTask& task);

private:
	Fsm mFsm;
	size_t mReInitial;
	Fsm mDetermined;
	TransitionTagTable mActions;
	bool mSimple;
};

class CountingFsmTask {
public:
	typedef Fsm::LettersTbl LettersTbl;

	virtual ~CountingFsmTask() {}

	void Connect(size_t from, size_t to, Char letter) {
		mNewFsm.Connect(from, to, letter);
	}

	typedef bool Result;

	static Result Success() {
		return true;
	}

	static Result Failure() {
		return false;
	}

	Fsm& Output() {
		return mNewFsm;
	}

	TransitionTagTable& Actions() {
		return mNewActions;
	}

protected:
	void ResizeOutput(size_t size) {
		mNewFsm.Resize(size);
		mNewActions.resize(size);
	}

private:
	Fsm mNewFsm;
	TransitionTagTable mNewActions;
};

class StateLessForMinimize {
public:
	StateLessForMinimize(const CountingFsm& fsm) : mFsm(fsm) {}

	bool operator()(size_t first, size_t second) const {
		for (auto&& lettersEl : mFsm.Letters()) {
			const auto letter = lettersEl.first;
			if (mFsm.Output(first, letter) != mFsm.Output(second, letter)) {
				return mFsm.Output(first, letter) < mFsm.Output(second, letter);
			}
		}
		return false;
	}

private:
	const CountingFsm& mFsm;
};


class CountingFsmMinimizeTask : public CountingFsmTask {
public:
	explicit CountingFsmMinimizeTask(const CountingFsm& fsm)
		: mFsm(fsm)
		, reversedTransitions(fsm.Determined().Size())
		, StateClass(fsm.Determined().Size())
		, Classes(0)
	{
		TMap<size_t, size_t, StateLessForMinimize> stateClassMap = TMap<size_t, size_t, StateLessForMinimize>(StateLessForMinimize(mFsm));
		for (size_t state = 0; state < mFsm.Determined().Size(); ++state) {
			if (stateClassMap.find(state) == stateClassMap.end()) {
				stateClassMap[state] = Classes++;
			}
			StateClass[state] = stateClassMap[state];
			reversedTransitions[state].resize(mFsm.Letters().Size());
		}

		for (size_t state = 0; state < mFsm.Determined().Size(); ++state) {
			TSet<ypair<Char, size_t>> usedTransitions;
			for (const auto& letter : mFsm.Letters()) {
				const auto destination = Next(state, letter.first);
				const auto letterId = letter.second.first;
				if (usedTransitions.find(ymake_pair(letterId, destination)) == usedTransitions.end()) {
					usedTransitions.insert(ymake_pair(letterId, destination));
					reversedTransitions[destination][letterId].push_back(state);
				}
			}
		}
	}

	TVector<size_t>& GetStateClass() { return StateClass; }

	size_t& GetClassesNumber() { return Classes; }

	size_t LettersCount() const {
		return mFsm.Letters().Size();
	}

	bool IsDetermined() const {
		return mFsm.Determined().IsDetermined();
	}

	size_t Size() const {
		return mFsm.Determined().Size();
	}

	const TVector<size_t>& Previous(size_t state, size_t letter) const {
		return reversedTransitions[state][letter];
	}

	void AcceptStates() {
		ResizeOutput(Classes);
		auto& newFsm = Output();
		auto& newActions = Actions();
		newFsm.SetFinal(0, false);

		// Unite equality classes into new states
		for (size_t from = 0; from != Size(); ++from) {
			const auto fromMinimized = StateClass[from];
			for (auto&& letter : mFsm.Letters()) {
				const auto representative = letter.first;
				const auto next = Next(from, representative);
				const auto nextMinimized = StateClass[next];
				Connect(fromMinimized, nextMinimized, representative);
				const auto outputs = mFsm.Output(from, representative);
				if (outputs) {
					newActions[fromMinimized][representative] = outputs;
				}
			}
			if (mFsm.Determined().IsFinal(from)) {
				newFsm.SetFinal(fromMinimized, true);
			}
		}
		newFsm.SetInitial(StateClass[mFsm.Determined().Initial()]);
		newFsm.SetIsDetermined(true);
	}

private:
	const CountingFsm& mFsm;
	TVector<TVector<TVector<size_t>>> reversedTransitions;
	TVector<size_t> StateClass;
	size_t Classes;

	size_t Next(size_t state, Char letter) const {
		const auto& tos = mFsm.Determined().Destinations(state, letter);
		Y_ASSERT(tos.size() == 1);
		return *tos.begin();
	}
};

typedef size_t RawState;
typedef ypair<RawState, unsigned long> TaggedState;
typedef TSet<TaggedState> StateGroup;

struct DeterminedState {
public:
	StateGroup matched;
	StateGroup unmatched;
	StateGroup separated;
	StateGroup lagging;
};

bool operator < (const DeterminedState& left, const DeterminedState& right) {
	auto asTuple = [](const DeterminedState& state) {
		return std::tie(state.matched, state.unmatched, state.separated, state.lagging);
	};

	return asTuple(left) < asTuple(right);
}

bool InvalidCharRange(const TVector<Char>& range) {
	for (const auto letter : range) {
		if (letter < MaxCharUnaligned && letter != 256) {
			return false;
		}
	}
	return true;
}

class BasicCountingFsmDetermineTask : public CountingFsmTask {
public:
	using CountingFsmTask::LettersTbl;
	typedef DeterminedState State;
	typedef TMap<State, size_t> InvStates;

	explicit BasicCountingFsmDetermineTask(const Fsm& fsm, RawState reInitial)
		: mFsm(fsm)
		, mReInitial{reInitial}
	{
		mDeadStates = fsm.DeadStates();
		for (auto&& letter : fsm.Letters()) {
			if (InvalidCharRange(letter.second.second)) {
				mInvalidLetters.insert(letter.first);
			}
		}
	}

	const LettersTbl& Letters() const {
		return mFsm.Letters();
	}

	State Initial() const {
		return State{StateGroup{}, InitialGroup(), StateGroup{}, StateGroup{}};
	}

	bool IsRequired(const State& state) const {
		Y_UNUSED(state);
		return true;
	}

	State Next(const State& state, Char letter) const {
		if (mInvalidLetters.count(letter) != 0) {
			AddAction(state, letter, CountingFsm::NotMatched);
			return Initial();
		}

		auto next = PrepareNextState(state, letter);
		AddAction(state, letter, CalculateTransitionTag(state, next));
		PostProcessNextState(next);
		NormalizeState(next);

		return next;
	}

	void AcceptStates(const TVector<State>& states)
	{
		ResizeOutput(states.size());
		auto& newFsm = Output();
		auto& newActions = Actions();
		newFsm.SetInitial(0);
		newFsm.SetIsDetermined(true);

		for (size_t ns = 0; ns < states.size(); ++ns) {
			const auto& state = states[ns];
			newFsm.SetFinal(ns, HasFinals(state.unmatched));

			auto outputIt = mActionByState.find(state);
			if (outputIt != mActionByState.end()) {
				newActions[ns].swap(outputIt->second);
			}
		}
	}

protected:
	void SplitDestinations(StateGroup& matched, StateGroup& unmatched, StateGroup& separated, const StateGroup& source, Char letter) const {
		for (const auto& state : source) {
			MakeTaggedStates(matched, unmatched, separated, mFsm.Destinations(state.first, letter), state.second);
			if (mFsm.IsFinal(state.first)) {
				// Implicit epsilon transitions from final states to reInitial after matching separator
				MakeTaggedStates(separated, separated, separated, mFsm.Destinations(mReInitial, letter), CountingFsm::Separated);
			}
		}
	}

	Action CalculateTransitionTagImpl(const State& dest) const {
		Action result = 0;
		if (!dest.matched.empty()) {
			result = AdvancedCountingScanner::IncrementAction;
		} else if (dest.unmatched.empty()) {
			if (!dest.separated.empty()) {
				for (const auto& state : dest.separated) {
					if (state.second == CountingFsm::Matched) {
						result = AdvancedCountingScanner::IncrementAction;
					}
				}
			} else {
				result = AdvancedCountingScanner::ResetAction;
				for (const auto& state : dest.lagging) {
					if (state.second != CountingFsm::NotMatched) {
						result |= AdvancedCountingScanner::IncrementAction;
					}
				}
			}
		}
		return result;
	}

	unsigned long TagsOfGroup(const StateGroup& group) const {
		unsigned long result = 0;
		for (const auto& state : group) {
			result |= state.second;
		}
		return result;
	}

	void SplitGroupByTag(StateGroup& matched, StateGroup& unmatched, StateGroup& separated, const StateGroup& source, bool useFsmTag) const {
		for (const auto& state : source) {
			auto tag = useFsmTag ? mFsm.Tag(state.first) : state.second;
			if (tag == CountingFsm::Matched) {
				matched.insert(state);
			} else if (tag == CountingFsm::Separated) {
				separated.insert(state);
			} else {
				unmatched.insert(state);
			}
		}
	}

	void UpdateLaggingStates(State& state, bool moveToLagging) const {
		if (!state.matched.empty()) {
			if (moveToLagging) {
				state.lagging.insert(state.unmatched.cbegin(), state.unmatched.cend());
				state.lagging.insert(state.separated.cbegin(), state.separated.cend());
			}
			state.unmatched.clear();
			state.separated.clear();
		}
		if (state.unmatched.empty() && !state.separated.empty()) {
			const auto unmatchedTags = TagsOfGroup(state.separated);
			if ((unmatchedTags & CountingFsm::Matched) && (unmatchedTags != CountingFsm::Matched)) {
				StateGroup separatedMatched;
				for (const auto& separatedState : state.separated) {
					if (separatedState.second == CountingFsm::Matched) {
						separatedMatched.insert(separatedState);
					} else if (moveToLagging) {
						state.lagging.insert(separatedState);
					}
				}
				state.separated.swap(separatedMatched);
			}
		}
	}

	void RemoveDuplicateLaggingStates(State& state) const {
		const auto statesToRemove = GetRawStates({state.matched, state.unmatched, state.separated}, 0);
		const auto unmatchedStatesToRemove = GetRawStates({state.lagging}, CountingFsm::NotMatched);

		StateGroup newLagging;
		for (const auto& taggedState : state.lagging) {
			if (statesToRemove.count(taggedState.first) == 0) {
				if (taggedState.second != CountingFsm::NotMatched || unmatchedStatesToRemove.count(taggedState.first) == 0) {
					newLagging.insert(taggedState);
				}
			}
		}
		state.lagging.swap(newLagging);
	}

	void RemoveDuplicateSeparatedStates(State& state) const {
		if (state.separated.empty()) {
			return;
		}
		const auto statesToRemove = GetRawStates({state.matched, state.unmatched}, 0);
		RemoveRawStates(state.separated, statesToRemove);
	}

	void NormalizeState(State& state) const {
		if (!state.matched.empty()) {
			Y_ASSERT(state.unmatched.empty());
			state.unmatched.swap(state.matched);
		}

		if (state.unmatched.empty() && !state.separated.empty()) {
			state.unmatched.swap(state.separated);
		}

		if (state.unmatched.empty() && !state.lagging.empty()) {
			State groups;
			SplitGroupByTag(groups.matched, groups.unmatched, groups.separated, state.lagging, false);
			if (!groups.matched.empty()) {
				state.unmatched.swap(groups.matched);
				state.separated.swap(groups.separated);
				state.lagging.swap(groups.unmatched);
			} else if (!groups.separated.empty()) {
				state.unmatched.swap(groups.separated);
				state.lagging.swap(groups.unmatched);
			} else {
				state.unmatched.swap(groups.unmatched);
				state.lagging.swap(groups.matched); // just clear
			}
		}
	}

private:
	virtual State PrepareNextState(const State& state, Char letter) const = 0;

	virtual void PostProcessNextState(State& next) const = 0;

	virtual Action CalculateTransitionTag(const State& source, const State& dest) const {
		Y_UNUSED(source);
		return CalculateTransitionTagImpl(dest);
	}

	virtual StateGroup InitialGroup() const {
		return StateGroup{TaggedState{mFsm.Initial(), CountingFsm::NotMatched}};
	}

	void AddAction(State from, Char letter, unsigned long value) const {
		if (!value) {
			return;
		}
		TransitionTagRow& row = mActionByState[from];
		row[letter] = value;
	}

	void MakeTaggedStates(StateGroup& matched, StateGroup& unmatched, StateGroup& separated, const Fsm::StatesSet& destinations, unsigned long sourceTag) const {
		for (const auto destState : destinations) {
			if (mDeadStates.count(destState) == 0) {
				const auto destTag = mFsm.Tag(destState);
				if (sourceTag != CountingFsm::Matched && destTag == CountingFsm::Matched) {
					matched.insert(ymake_pair(destState, destTag));
				} else if (sourceTag == CountingFsm::Separated || destTag == CountingFsm::Separated) {
					separated.insert(ymake_pair(destState, CountingFsm::Separated));
				} else {
					unmatched.insert(ymake_pair(destState, sourceTag));
				}
			}
		}
	}

	bool HasFinals(const StateGroup& states) const {
		for (const auto& state : states) {
			if (mFsm.IsFinal(state.first)) {
				return true;
			}
		}
		return false;
	}

	Fsm::StatesSet GetRawStates(const TVector<std::reference_wrapper<const StateGroup>> groups, unsigned long excludedTags) const {
		Fsm::StatesSet result;
		for (const auto& group : groups) {
			for (const auto& taggedState : group.get()) {
				if (!(taggedState.second & excludedTags)) {
					result.insert(taggedState.first);
				}
			}
		}
		return result;
	}

	void RemoveRawStates(StateGroup& group, const Fsm::StatesSet& states) const {
		StateGroup removing;
		for (const auto& taggedState : group) {
			if (states.count(taggedState.first) != 0) {
				removing.insert(taggedState);
			}
		}
		for (const auto& taggedState : removing) {
			group.erase(taggedState);
		}
	}

private:
	const Fsm& mFsm;
	RawState mReInitial;
	Fsm::StatesSet mDeadStates;
	TSet<Char> mInvalidLetters;

	mutable TMap<State, TransitionTagRow> mActionByState;
};

class CountingFsmDetermineTask : public BasicCountingFsmDetermineTask {
public:
	using BasicCountingFsmDetermineTask::State;
	using BasicCountingFsmDetermineTask::LettersTbl;
	using BasicCountingFsmDetermineTask::InvStates;

	explicit CountingFsmDetermineTask(const Fsm& fsm, RawState reInitial)
		: BasicCountingFsmDetermineTask{fsm, reInitial}
	{}

private:
	State PrepareNextState(const State& state, Char letter) const override {
		State next;
		SplitDestinations(next.matched, next.unmatched, next.separated, state.unmatched, letter);
		SplitDestinations(next.separated, next.separated, next.separated, state.separated, letter);
		SplitDestinations(next.lagging, next.lagging, next.lagging, state.lagging, letter);
		return next;
	}

	void PostProcessNextState(State& next) const override {
		UpdateLaggingStates(next, true);
		RemoveDuplicateLaggingStates(next);
		RemoveDuplicateSeparatedStates(next);
	}
};

class SimpleCountingFsmDetermineTask : public BasicCountingFsmDetermineTask {
public:
	using BasicCountingFsmDetermineTask::State;
	using BasicCountingFsmDetermineTask::LettersTbl;
	using BasicCountingFsmDetermineTask::InvStates;

	static constexpr unsigned long MixedTags = CountingFsm::Separated | CountingFsm::Matched;

	SimpleCountingFsmDetermineTask(const Fsm& fsm, RawState reInitial)
		: BasicCountingFsmDetermineTask{fsm, reInitial}
		, mStartState{reInitial, CountingFsm::NotMatched}
	{}

private:
	State PrepareNextState(const State& state, Char letter) const override {
		State next;
		auto from = state;
		const auto fromIsEmpty = IsEmptyState(from);
		if (fromIsEmpty) {
			from.unmatched.insert(mStartState);
		}
		Y_ASSERT(IsValidState(from));

		SplitDestinations(next.matched, next.unmatched, next.separated, from.unmatched, letter);
		if (next.matched.empty() && !next.separated.empty()) {
			if (next.unmatched.empty()) {
				SplitSeparatedByFsmTag(next);
				if (next.separated.size() > 1) {
					RemoveDuplicateSeparatedStates(next);
				}
				if (next.unmatched.empty()) {
					next.unmatched.swap(next.separated);
				}
			} else {
				ChooseOneSeparatedState(next);
			}
		}
		if (next.matched.empty() && next.separated.empty() && !from.separated.empty()) {
			if (!next.unmatched.empty()) {
				ChooseOneDestState(next.separated, from.separated, letter);
			} else {
				SplitDestinations(next.matched, next.unmatched, next.separated, from.separated, letter);
				if (next.matched.empty() && !next.separated.empty()) {
					SplitSeparatedByFsmTag(next);
				}
			}
			ChooseOneSeparatedState(next);
		}
		if (!fromIsEmpty && IsEmptyState(next)) {
			ChooseOneDestState(next.lagging, StateGroup{mStartState}, letter);
		}

		return next;
	}

	void PostProcessNextState(State& next) const override {
		if (!next.lagging.empty()) {
			next.unmatched.swap(next.lagging);
		}
		UpdateLaggingStates(next, false);
		RemoveDuplicateSeparatedStates(next);
	}

	Action CalculateTransitionTag(const State& source, const State& dest) const override {
		Action tag = CalculateTransitionTagImpl(dest);
		if (!((TagsOfGroup(source.unmatched) | TagsOfGroup(source.separated)) & MixedTags)) {
			tag &= AdvancedCountingScanner::IncrementAction;
		}
		return tag;
	}

	StateGroup InitialGroup() const override {
		return StateGroup{};
	}

	bool IsEmptyState(const State& state) const {
		return state.matched.empty() && state.unmatched.empty() && state.separated.empty() && state.lagging.empty();
	}

	bool IsValidState(const State& state) const {
		return state.matched.empty() && state.unmatched.size() <= 1 && state.separated.size() <= 1 && state.lagging.empty();
	}

	void SplitSeparatedByFsmTag(State& state) const {
		Y_ASSERT(state.unmatched.empty());
		StateGroup separated;
		separated.swap(state.separated);
		SplitGroupByTag(state.matched, state.unmatched, state.separated, separated, true);
	}

	void ChooseOneDestState(StateGroup& dest, const StateGroup& source, Char letter) const {
		State destState;
		SplitDestinations(destState.matched, destState.unmatched, destState.separated, source, letter);
		if (!destState.matched.empty()) {
			dest.swap(destState.matched);
		} else if (!destState.separated.empty()) {
			dest.swap(destState.separated);
		} else if (!destState.unmatched.empty()) {
			dest.swap(destState.unmatched);
		}
	}

	void ChooseOneSeparatedState(State& state) const {
		if (state.separated.size() <= 1) {
			return;
		}
		RemoveDuplicateSeparatedStates(state);
		State splitted;
		SplitGroupByTag(splitted.matched, splitted.unmatched, splitted.separated, state.separated, true);
		if (!splitted.separated.empty()) {
			state.separated.swap(splitted.separated);
		} else if (!splitted.matched.empty()) {
			state.separated.swap(splitted.matched);
		}
	}

private:
	TaggedState mStartState;
};

bool CountingFsm::Determine() {
	CountingFsmDetermineTask task{mFsm, mReInitial};
	size_t maxSize = mFsm.Size() * 4096;
	if (Pire::Impl::Determine(task, maxSize)) {
		SwapTaskOutputs(task);
		mSimple = false;
	} else {
		SimpleCountingFsmDetermineTask simpleTask{mFsm, mReInitial};
		if (Pire::Impl::Determine(simpleTask, std::numeric_limits<size_t>::max())) {
			SwapTaskOutputs(simpleTask);
			mSimple = true;
		} else {
			return false;
		}
	}
	return true;
}

void CountingFsm::Minimize() {
	CountingFsmMinimizeTask task{*this};
	Pire::Impl::Minimize(task);
	SwapTaskOutputs(task);
}

void CountingFsm::SwapTaskOutputs(CountingFsmTask& task) {
	task.Output().Swap(mDetermined);
	task.Actions().swap(mActions);
}

}

namespace {
	Pire::Fsm FsmForDot() { Pire::Fsm f; f.AppendDot(); return f; }
	Pire::Fsm FsmForChar(Pire::Char c) { Pire::Fsm f; f.AppendSpecial(c); return f; }
}

CountingScanner::CountingScanner(const Fsm& re, const Fsm& sep)
{
	Fsm res = re;
	res.Surround();
	Fsm sep_re = ((sep & ~res) /* | Fsm()*/) + re;
	sep_re.Determine();

	Fsm dup = sep_re;
	for (size_t i = 0; i < dup.Size(); ++i)
		dup.SetTag(i, Matched);
	size_t oldsize = sep_re.Size();
	sep_re.Import(dup);
	for (Fsm::FinalTable::const_iterator i = sep_re.Finals().begin(), ie = sep_re.Finals().end(); i != ie; ++i)
		if (*i < oldsize)
			sep_re.Connect(*i, oldsize + *i);

	sep_re |= (FsmForDot() | FsmForChar(Pire::BeginMark) | FsmForChar(Pire::EndMark));

	// Make a full Cartesian product of two sep_res
	sep_re.Determine();
	sep_re.Unsparse();
	TSet<size_t> dead = sep_re.DeadStates();

	PIRE_IFDEBUG(Cdbg << "=== Original FSM ===" << Endl << sep_re << ">>> " << sep_re.Size() << " states, dead: [" << Join(dead.begin(), dead.end(), ", ") << "]" << Endl);

	Fsm sq;

	typedef ypair<size_t, size_t> NewState;
	TVector<NewState> states;
	TMap<NewState, size_t> invstates;

	states.push_back(NewState(sep_re.Initial(), sep_re.Initial()));
	invstates.insert(ymake_pair(states.back(), states.size() - 1));

	// TODO: this loop reminds me a general determination task...
	for (size_t curstate = 0; curstate < states.size(); ++curstate) {

		unsigned long tag = sep_re.Tag(states[curstate].first);
		if (tag)
			sq.SetTag(curstate, tag);
		sq.SetFinal(curstate, sep_re.IsFinal(states[curstate].first));

		PIRE_IFDEBUG(Cdbg << "State " << curstate << " = (" << states[curstate].first << ", " << states[curstate].second << ")" << Endl);
		for (Fsm::LettersTbl::ConstIterator lit = sep_re.Letters().Begin(), lie = sep_re.Letters().End(); lit != lie; ++lit) {

			Char letter = lit->first;

			const Fsm::StatesSet& mr = sep_re.Destinations(states[curstate].first, letter);
			const Fsm::StatesSet& br = sep_re.Destinations(states[curstate].second, letter);

			if (mr.size() != 1)
				Y_ASSERT(!"Wrong transition size for main");
			if (br.size() != 1)
				Y_ASSERT(!"Wrong transition size for backup");

			NewState ns(*mr.begin(), *br.begin());
			PIRE_IFDEBUG(NewState savedNs = ns);
			unsigned long outputs = 0;

			PIRE_IFDEBUG(ystring dbgout);
			if (dead.find(ns.first) != dead.end()) {
				PIRE_IFDEBUG(dbgout = ((sep_re.Tag(ns.first) & Matched) ? ", ++cur" : ", max <- cur"));
				outputs = DeadFlag | (sep_re.Tag(ns.first) & Matched);
				ns.first = ns.second;
			}
			if (sep_re.IsFinal(ns.first) || (sep_re.IsFinal(ns.second) && !(sep_re.Tag(ns.first) & Matched)))
				ns.second = sep_re.Initial();

			PIRE_IFDEBUG(if (ns != savedNs) Cdbg << "Diverted transition to (" << savedNs.first << ", " << savedNs.second << ") on " << (char) letter << " to (" << ns.first << ", " << ns.second << ")" << dbgout << Endl);

			TMap<NewState, size_t>::iterator nsi = invstates.find(ns);
			if (nsi == invstates.end()) {
				PIRE_IFDEBUG(Cdbg << "New state " << states.size() << " = (" << ns.first << ", " << ns.second << ")" << Endl);
				states.push_back(ns);
				nsi = invstates.insert(ymake_pair(states.back(), states.size() - 1)).first;
				sq.Resize(states.size());
			}

			for (TVector<Char>::const_iterator li = lit->second.second.begin(), le = lit->second.second.end(); li != le; ++li)
			sq.Connect(curstate, nsi->second, *li);
			if (outputs)
				sq.SetOutput(curstate, nsi->second, outputs);
		}
	}

	sq.Determine();

	PIRE_IFDEBUG(Cdbg << "=== FSM ===" << Endl << sq << Endl);
	Init(sq.Size(), sq.Letters(), sq.Initial(), 1);
	BuildScanner(sq, *this);
}

namespace Impl {
template <class AdvancedScanner>
AdvancedScanner MakeAdvancedCountingScanner(const Fsm& re, const Fsm& sep, bool* simple) {
	Impl::CountingFsm countingFsm{re, sep};
	if (!countingFsm.Determine()) {
		throw Error("regexp pattern too complicated");
	}
	countingFsm.Minimize();
	if (simple) {
		*simple = countingFsm.Simple();
	}

	const auto& determined = countingFsm.Determined();
	const auto& letters = countingFsm.Letters();

	AdvancedScanner scanner;
	scanner.Init(determined.Size(), letters, determined.Initial(), 1);
	for (size_t from = 0; from != determined.Size(); ++from) {
		for (auto&& lettersEl : letters) {
			const auto letter = lettersEl.first;
			const auto& tos = determined.Destinations(from, letter);
			Y_ASSERT(tos.size() == 1);
			scanner.SetJump(from, letter, *tos.begin(), scanner.RemapAction(countingFsm.Output(from, letter)));
		}
	}
	return scanner;
}
}  // namespace Impl

AdvancedCountingScanner::AdvancedCountingScanner(const Fsm& re, const Fsm& sep, bool* simple)
	: AdvancedCountingScanner(Impl::MakeAdvancedCountingScanner<AdvancedCountingScanner>(re, sep, simple))
{
}

NoGlueLimitCountingScanner::NoGlueLimitCountingScanner(const Fsm& re, const Fsm& sep, bool* simple)
	: NoGlueLimitCountingScanner(Impl::MakeAdvancedCountingScanner<NoGlueLimitCountingScanner>(re, sep, simple))
{
}


namespace Impl {

template<class Scanner>
class CountingScannerGlueTask: public ScannerGlueCommon<Scanner> {
public:
	using typename ScannerGlueCommon<Scanner>::State;
	using TAction = typename Scanner::Action;
	using InternalState = typename Scanner::InternalState;
	typedef TMap<State, size_t> InvStates;

	CountingScannerGlueTask(const Scanner& lhs, const Scanner& rhs)
		: ScannerGlueCommon<Scanner>(lhs, rhs, LettersEquality<Scanner>(lhs.m_letters, rhs.m_letters))
	{
	}

	void AcceptStates(const TVector<State>& states)
	{
		States = states;
		this->SetSc(THolder<Scanner>(new Scanner));
		this->Sc().Init(states.size(), this->Letters(), 0, this->Lhs().RegexpsCount() + this->Rhs().RegexpsCount());

		for (size_t i = 0; i < states.size(); ++i)
			this->Sc().SetTag(i, this->Lhs().m_tags[this->Lhs().StateIdx(states[i].first)] | (this->Rhs().m_tags[this->Rhs().StateIdx(states[i].second)] << 3));
	}

	void Connect(size_t from, size_t to, Char letter)
	{
		this->Sc().SetJump(from, letter, to,
			Action(this->Lhs(), States[from].first, letter) | (Action(this->Rhs(), States[from].second, letter) << this->Lhs().RegexpsCount()));
	}

protected:
	TVector<State> States;
	TAction Action(const Scanner& sc, InternalState state, Char letter) const
	{
		size_t state_index = sc.StateIdx(state);
		size_t transition_index = sc.TransitionIndex(state_index, letter);
		const auto& tr = sc.m_jumps[transition_index];
		return tr.action;
	}
};

class NoGlueLimitCountingScannerGlueTask : public CountingScannerGlueTask<NoGlueLimitCountingScanner> {
public:
	using ActionIndex = NoGlueLimitCountingScanner::ActionIndex;
	struct TGlueAction {
		TVector<ActionIndex> resets;
		TVector<ActionIndex> increments;
		bool operator<(const TGlueAction& rhs) const {
			return std::tie(increments, resets) < std::tie(rhs.increments, rhs.resets);
		}
	};
	using TGlueMap = TMap<TGlueAction, ActionIndex>;

	NoGlueLimitCountingScannerGlueTask(const NoGlueLimitCountingScanner& lhs, const NoGlueLimitCountingScanner& rhs)
		: CountingScannerGlueTask(lhs, rhs)
	{
	}

	void Connect(size_t from, size_t to, Char letter)
	{
		TGlueAction glue_action;
		this->Lhs().GetActions(Action(this->Lhs(), States[from].first, letter), 0,
							   std::back_inserter(glue_action.resets), std::back_inserter(glue_action.increments));
		this->Rhs().GetActions(Action(this->Rhs(), States[from].second, letter), this->Lhs().RegexpsCount(),
							   std::back_inserter(glue_action.resets), std::back_inserter(glue_action.increments));
		Y_ASSERT(
			std::is_sorted(glue_action.increments.begin(), glue_action.increments.end()) &&
			std::is_sorted(glue_action.resets.begin(), glue_action.resets.end())
		);

		if (glue_action.increments.empty() && glue_action.resets.empty()) {
			this->Sc().SetJump(from, letter, to, 0);
			return;
		}

		auto action_iter = glue_map_.find(glue_action);
		if (action_iter == glue_map_.end()) {
			glue_map_[glue_action] = glue_actions_.size();
			for (const auto& ids : {glue_action.resets, glue_action.increments}) {
				glue_actions_.push_back(ids.size());
				std::copy(ids.begin(), ids.end(), std::back_inserter(glue_actions_));
			}
		}

		this->Sc().SetJump(from, letter, to, glue_map_[glue_action]);
	}

	// Return type is same as in parent class
	// TODO: Maybe return by value to use move semantic?
	const NoGlueLimitCountingScanner& Success()
	{
		glue_actions_[0] = glue_actions_.size();
		Sc().AcceptActions(glue_actions_);
		return Sc();
	}

private:
	TGlueMap glue_map_;
	TVector<ActionIndex> glue_actions_ = {1};
};


}

CountingScanner CountingScanner::Glue(const CountingScanner& lhs, const CountingScanner& rhs, size_t maxSize /* = 0 */)
{
	if (lhs.RegexpsCount() + rhs.RegexpsCount() > MAX_RE_COUNT) {
		return CountingScanner();
	}
	static constexpr size_t DefMaxSize = 250000;
	Impl::CountingScannerGlueTask<CountingScanner> task(lhs, rhs);
	return Impl::Determine(task, maxSize ? maxSize : DefMaxSize);
}

AdvancedCountingScanner AdvancedCountingScanner::Glue(const AdvancedCountingScanner& lhs, const AdvancedCountingScanner& rhs, size_t maxSize /* = 0 */)
{
	if (lhs.RegexpsCount() + rhs.RegexpsCount() > MAX_RE_COUNT) {
		return AdvancedCountingScanner();
	}
	static constexpr size_t DefMaxSize = 250000;
	Impl::CountingScannerGlueTask<AdvancedCountingScanner> task(lhs, rhs);
	return Impl::Determine(task, maxSize ? maxSize : DefMaxSize);
}

NoGlueLimitCountingScanner NoGlueLimitCountingScanner::Glue(const NoGlueLimitCountingScanner& lhs, const NoGlueLimitCountingScanner& rhs, size_t maxSize /* = 0 */)
{
	static constexpr size_t DefMaxSize = 250000;
	Impl::NoGlueLimitCountingScannerGlueTask task(lhs, rhs);
	return Impl::Determine(task, maxSize ? maxSize : DefMaxSize);
}

// Should Save(), Load() and Mmap() functions return stream/pointer in aligned state?
// Now they don't because tests don't require it.
void NoGlueLimitCountingScanner::Save(yostream* s) const {
	Y_ASSERT(!AdvancedScannerCompatibilityMode);
	LoadedScanner::Save(s, ScannerIOTypes::NoGlueLimitCountingScanner);
	if (Actions) {
		SavePodArray(s, Actions, *Actions);
	} else {
		const ActionIndex zeroSize = 0;
		SavePodType(s, zeroSize);
	}
}

void NoGlueLimitCountingScanner::Load(yistream* s) {
	ui32 type;
	LoadedScanner::Load(s, &type);
	ActionIndex actionsSize;
	if (type == ScannerIOTypes::NoGlueLimitCountingScanner) {
		LoadPodType(s, actionsSize);

		if (actionsSize == 0) {
			ActionsBuffer.reset();
			Actions = nullptr;
		} else {
			ActionsBuffer = TActionsBuffer(new ActionIndex[actionsSize]);
			ActionsBuffer[0] = actionsSize;
			LoadPodArray(s, &ActionsBuffer[1], actionsSize - 1);
			Actions = ActionsBuffer.get();
		}
	} else {
		Y_ASSERT(type == ScannerIOTypes::LoadedScanner);
		AdvancedScannerCompatibilityMode = true;
	}
}

const void* NoGlueLimitCountingScanner::Mmap(const void* ptr, size_t size) {
	NoGlueLimitCountingScanner scanner;
	ui32 type;
	auto p = static_cast<const size_t*> (scanner.LoadedScanner::Mmap(ptr, size, &type));

	if (type == ScannerIOTypes::NoGlueLimitCountingScanner) {
		scanner.Actions = reinterpret_cast<const ActionIndex*>(p);
		if (*scanner.Actions == 0) {
			scanner.Actions = nullptr;
			Impl::AdvancePtr(p, size, sizeof(ActionIndex));
		} else {
			Impl::AdvancePtr(p, size, *scanner.Actions * sizeof(ActionIndex));
		}
	} else {
		Y_ASSERT(type == ScannerIOTypes::LoadedScanner);
		scanner.AdvancedScannerCompatibilityMode = true;
	}
	Swap(scanner);
	return static_cast<const void*>(p);
}

}
