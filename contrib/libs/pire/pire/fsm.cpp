/*
 * fsm.cpp -- the implementation of the FSM class.
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


#include <algorithm>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <iterator>
#include <numeric>
#include <queue>
#include <utility>

#include <iostream>
#include <stdio.h>
#include <contrib/libs/pire/pire/stub/lexical_cast.h>

#include "fsm.h"
#include "vbitset.h"
#include "partition.h"
#include "determine.h"
#include "minimize.h"
#include "platform.h"

namespace Pire {

ystring CharDump(Char c)
{
	char buf[8];
	if (c == '"')
		return ystring("\\\"");
	else if (c == '[' || c == ']' || c == '-' || c == '^') {
		snprintf(buf, sizeof(buf)-1, "\\\\%c", c);
		return ystring(buf);
	} else if (c >= 32 && c < 127)
		return ystring(1, static_cast<char>(c));
	else if (c == '\n')
		return ystring("\\\\n");
	else if (c == '\t')
		return ystring("\\\\t");
	else if (c == '\r')
		return ystring("\\\\r");
	else if (c < 256) {
		snprintf(buf, sizeof(buf)-1, "\\\\%03o", static_cast<int>(c));
		return ystring(buf);
	} else if (c == Epsilon)
		return ystring("<Epsilon>");
	else if (c == BeginMark)
		return ystring("<Begin>");
	else if (c == EndMark)
		return ystring("<End>");
	else
		return ystring("<?" "?" "?>");
}

void Fsm::DumpState(yostream& s, size_t state) const
{
	// Fill in a 'row': Q -> exp(V) (for current state)
	TVector< ybitset<MaxChar> > row(Size());
	for (auto&& transition : m_transitions[state])
		for (auto&& transitionState : transition.second) {
			if (transitionState >= Size()) {
				std::cerr << "WTF?! Transition from " << state << " on letter " << transition.first << " leads to non-existing state " << transitionState << "\n";
				Y_ASSERT(false);
			}
			if (Letters().Contains(transition.first)) {
				const TVector<Char>& letters = Letters().Klass(Letters().Representative(transition.first));
				for (auto&& letter : letters)
					row[transitionState].set(letter);
			} else
				row[transitionState].set(transition.first);
		}

	bool statePrinted = false;
	// Display each destination state
	for (auto rit = row.begin(), rie = row.end(); rit != rie; ++rit) {
		unsigned begin = 0, end = 0;

		ystring delimiter;
		ystring label;
		if (rit->test(Epsilon)) {
			label += delimiter + CharDump(Epsilon);
			delimiter = " ";
		}
		if (rit->test(BeginMark)) {
			label += delimiter + CharDump(BeginMark);
			delimiter = " ";
		}
		if (rit->test(EndMark)) {
			label += delimiter + CharDump(EndMark);
			delimiter = " ";
		}
		unsigned count = 0;
		for (unsigned i = 0; i < 256; ++i)
			if (rit->test(i))
				++count;
		if (count != 0 && count != 256) {
			label += delimiter + "[";
			bool complementary = (count > 128);
			if (count > 128)
				label += "^";
			while (begin < 256) {
				for (begin = end; begin < 256 && (rit->test(begin) == complementary); ++begin)
					;
				for (end = begin; end < 256 && (rit->test(end) == !complementary); ++end)
					;
				if (begin + 1 == end) {
					label += CharDump(begin);
					delimiter = " ";
				} else if (begin != end) {
					label += CharDump(begin) + "-" + (CharDump(end-1));
					delimiter = " ";
				}
			}
			label += "]";
			delimiter = " ";
		} else if (count == 256) {
			label += delimiter + ".";
			delimiter = " ";
		}
		if (!label.empty()) {
			if (!statePrinted) {
				s << "    " << state << "[shape=\"" << (IsFinal(state) ? "double" : "") << "circle\",label=\"" << state;
				auto ti = tags.find(state);
				if (ti != tags.end())
					s << " (tags: " << ti->second << ")";
				s << "\"]\n";
				if (Initial() == state)
					s << "    \"initial\" -> " << state << '\n';
				statePrinted = true;
			}
			s << "    " << state << " -> " << std::distance(row.begin(), rit) << "[label=\"" << label;

			// Display outputs
			auto oit = outputs.find(state);
			if (oit != outputs.end()) {
				auto oit2 = oit->second.find(std::distance(row.begin(), rit));
				if (oit2 == oit->second.end())
					;
				else {
					TVector<int> payload;
					for (unsigned i = 0; i < sizeof(oit2->second) * 8; ++i)
						if (oit2->second & (1ul << i))
							payload.push_back(i);
					if (!payload.empty())
						s << " (outputs: " << Join(payload.begin(), payload.end(), ", ") << ")";
				}
			}

			s << "\"]\n";
		}
	}

	if (statePrinted)
		s << '\n';
}

void Fsm::DumpTo(yostream& s, const ystring& name) const
{
	s << "digraph {\n    \"initial\"[shape=\"plaintext\",label=\"" << name << "\"]\n\n";
	for (size_t state = 0; state < Size(); ++state) {
		DumpState(s, state);
	}
	s << "}\n\n";
}

yostream& operator << (yostream& s, const Fsm& fsm) { fsm.DumpTo(s); return s; }


namespace {
	template<class Vector> void resizeVector(Vector& v, size_t s) { v.resize(s); }
}

Fsm::Fsm():
	m_transitions(1),
	initial(0),
	letters(m_transitions),
	m_sparsed(false),
	determined(false),
	isAlternative(false)
{
	m_final.insert(0);
}

Fsm Fsm::MakeFalse()
{
	Fsm f;
	f.SetFinal(0, false);
	return f;
}

Char Fsm::Translate(Char c) const
{
	if (!m_sparsed || c == Epsilon)
		return c;
	else
		return Letters().Representative(c);
}

bool Fsm::Connected(size_t from, size_t to, Char c) const
{
	auto it = m_transitions[from].find(Translate(c));
	return (it != m_transitions[from].end() && it->second.find(to) != it->second.end());
}

bool Fsm::Connected(size_t from, size_t to) const
{
	for (auto i = m_transitions[from].begin(), ie = m_transitions[from].end(); i != ie; ++i)
		if (i->second.find(to) != i->second.end())
			return true;
	return false;
}

const Fsm::StatesSet& Fsm::Destinations(size_t from, Char c) const
{
	auto i = m_transitions[from].find(Translate(c));
	return (i != m_transitions[from].end()) ? i->second : DefaultValue<StatesSet>();
}

TSet<Char> Fsm::OutgoingLetters(size_t state) const
{
	TSet<Char> ret;
	for (auto&& i : m_transitions[state])
		ret.insert(i.first);
	return ret;
}

size_t Fsm::Resize(size_t newSize)
{
	size_t ret = Size();
	m_transitions.resize(newSize);
	return ret;
}

void Fsm::Swap(Fsm& fsm)
{
	DoSwap(m_transitions, fsm.m_transitions);
	DoSwap(initial, fsm.initial);
	DoSwap(m_final, fsm.m_final);
	DoSwap(letters, fsm.letters);
	DoSwap(determined, fsm.determined);
	DoSwap(outputs, fsm.outputs);
	DoSwap(tags, fsm.tags);
	DoSwap(isAlternative, fsm.isAlternative);
}

void Fsm::SetFinal(size_t state, bool final)
{
	if (final)
		m_final.insert(state);
	else
		m_final.erase(state);
}

Fsm& Fsm::AppendDot()
{
	Resize(Size() + 1);
	for (size_t letter = 0; letter != (1 << (sizeof(char)*8)); ++letter)
		ConnectFinal(Size() - 1, letter);
	ClearFinal();
	SetFinal(Size() - 1, true);
	determined = false;
    return *this;
}

Fsm& Fsm::Append(char c)
{
	Resize(Size() + 1);
	ConnectFinal(Size() - 1, static_cast<unsigned char>(c));
	ClearFinal();
	SetFinal(Size() - 1, true);
	determined = false;
    return *this;
}

Fsm& Fsm::Append(const ystring& str)
{
    for (auto&& i : str)
        Append(i);
    return *this;
}

Fsm& Fsm::AppendSpecial(Char c)
{
	Resize(Size() + 1);
	ConnectFinal(Size() - 1, c);
	ClearFinal();
	SetFinal(Size() - 1, true);
	determined = false;
    return *this;
}

Fsm& Fsm::AppendStrings(const TVector<ystring>& strings)
{
	for (auto&& i : strings)
		if (i.empty())
			throw Error("None of strings passed to appendStrings() can be empty");

	Resize(Size() + 1);
	size_t end = Size() - 1;

	// A local transitions table: (oldstate, char) -> newstate.
	// Valid for all letters in given strings except final ones,
	// which are always connected to the end state.

	// NB: since each FSM contains at least one state,
	// state #0 cannot appear in LTRs. Thus we can use this
	// criteria to test whether a transition has been created or not.
	typedef ypair<size_t, char> Transition;
	TMap<char, size_t> startLtr;
	TMap<Transition, size_t> ltr;

	// A presense of a transition in this set indicates that
	// a that transition already points somewhere (either to end
	// or somewhere else). Another attempt to create such transition
	// will clear `determined flag.
	TSet<Transition> usedTransitions;
	TSet<char> usedFirsts;

	for (const auto& str : strings) {
		if (str.size() > 1) {

			// First letter: all previously final states are connected to the new state
			size_t& firstJump = startLtr[str[0]];
			if (!firstJump) {
				firstJump = Resize(Size() + 1);
				ConnectFinal(firstJump, static_cast<unsigned char>(str[0]));
				determined = determined && (usedFirsts.find(str[0]) != usedFirsts.end());
			}

			// All other letters except last one
			size_t state = firstJump;
			for (auto cit = str.begin() + 1, cie = str.end() - 1; cit != cie; ++cit) {
				size_t& newState = ltr[ymake_pair(state, *cit)];
				if (!newState) {
					newState = Resize(Size() + 1);
					Connect(state, newState, static_cast<unsigned char>(*cit));
					determined = determined && (usedTransitions.find(ymake_pair(state, *cit)) != usedTransitions.end());
				}
				state = newState;
			}

			// The last letter: connect the current state to end
			unsigned char last = static_cast<unsigned char>(*(str.end() - 1));
			Connect(state, end, last);
			determined = determined && (usedTransitions.find(ymake_pair(state, last)) != usedTransitions.end());

		} else {
			// The single letter: connect all the previously final states to end
			ConnectFinal(end, static_cast<unsigned char>(str[0]));
			determined = determined && (usedFirsts.find(str[0]) != usedFirsts.end());
		}
	}

	ClearFinal();
	SetFinal(end, true);
    return *this;
}

void Fsm::Import(const Fsm& rhs)
{
//     PIRE_IFDEBUG(LOG_DEBUG("fsm") << "Importing");
//     PIRE_IFDEBUG(LOG_DEBUG("fsm") << "=== Left-hand side ===\n" << *this);
//     PIRE_IFDEBUG(LOG_DEBUG("fsm") << "=== Right-hand side ===\n" << rhs);

	size_t oldsize = Resize(Size() + rhs.Size());

	for (auto&& outer : m_transitions) {
		for (auto&& letter : letters) {
			auto targets = outer.find(letter.first);
			if (targets == outer.end())
				continue;
			for (auto&& character : letter.second.second)
				if (character != letter.first)
					outer.insert(ymake_pair(character, targets->second));
		}
	}

	auto dest = m_transitions.begin() + oldsize;
	for (auto outer = rhs.m_transitions.begin(), outerEnd = rhs.m_transitions.end(); outer != outerEnd; ++outer, ++dest) {
		for (auto&& inner : *outer) {
			TSet<size_t> targets;
			std::transform(inner.second.begin(), inner.second.end(), std::inserter(targets, targets.begin()),
				std::bind2nd(std::plus<size_t>(), oldsize));
			dest->insert(ymake_pair(inner.first, targets));
		}

		for (auto&& letter : rhs.letters) {
			auto targets = dest->find(letter.first);
			if (targets == dest->end())
				continue;
			for (auto&& character : letter.second.second)
				if (character != letter.first)
					dest->insert(ymake_pair(character, targets->second));
		}
	}

	// Import outputs
	for (auto&& output : rhs.outputs) {
		auto& dest = outputs[output.first + oldsize];
		for (auto&& element : output.second)
			dest.insert(ymake_pair(element.first + oldsize, element.second));
	}

	// Import tags
	for (auto&& tag : rhs.tags)
		tags.insert(ymake_pair(tag.first + oldsize, tag.second));

	letters = LettersTbl(LettersEquality(m_transitions));
}

void Fsm::Connect(size_t from, size_t to, Char c /* = Epsilon */)
{
	m_transitions[from][c].insert(to);
	ClearHints();
}

void Fsm::ConnectFinal(size_t to, Char c /* = Epsilon */)
{
	for (auto&& final : m_final)
		Connect(final, to, c);
	ClearHints();
}

void Fsm::Disconnect(size_t from, size_t to, Char c)
{
	auto i = m_transitions[from].find(c);
	if (i != m_transitions[from].end())
		i->second.erase(to);
	ClearHints();
}

void Fsm::Disconnect(size_t from, size_t to)
{
	for (auto&& i : m_transitions[from])
		i.second.erase(to);
	ClearHints();
}

unsigned long Fsm::Output(size_t from, size_t to) const
{
	auto i = outputs.find(from);
	if (i == outputs.end())
		return 0;
	auto j = i->second.find(to);
	if (j == i->second.end())
		return 0;
	else
		return j->second;
}

Fsm& Fsm::operator += (const Fsm& rhs)
{
	size_t lhsSize = Size();
	Import(rhs);

	const TransitionRow& row = m_transitions[lhsSize + rhs.initial];

	for (auto&& outer : row)
		for (auto&& inner : outer.second)
			ConnectFinal(inner, outer.first);

	auto out = rhs.outputs.find(rhs.initial);
	if (out != rhs.outputs.end())
		for (auto&& toAndOutput : out->second) {
			for (auto&& final : m_final)
				outputs[final].insert(ymake_pair(toAndOutput.first + lhsSize, toAndOutput.second));
		}

	ClearFinal();
	for (auto&& letter : rhs.m_final)
		SetFinal(letter + lhsSize, true);
	determined = false;

	ClearHints();
	PIRE_IFDEBUG(Cdbg << "=== After addition ===" << Endl << *this << Endl);

	return *this;
}

Fsm& Fsm::operator |= (const Fsm& rhs)
{
	size_t lhsSize = Size();

	Import(rhs);
	for (auto&& final : rhs.m_final)
		m_final.insert(final + lhsSize);

	if (!isAlternative && !rhs.isAlternative) {
		Resize(Size() + 1);
		Connect(Size() - 1, initial);
		Connect(Size() - 1, lhsSize + rhs.initial);
		initial = Size() - 1;
	} else if (isAlternative && !rhs.isAlternative) {
		Connect(initial, lhsSize + rhs.initial, Epsilon);
	} else if (!isAlternative && rhs.isAlternative) {
		Connect(lhsSize + rhs.initial, initial, Epsilon);
		initial = rhs.initial + lhsSize;
	} else if (isAlternative && rhs.isAlternative) {
		const StatesSet& tos = rhs.Destinations(rhs.initial, Epsilon);
		for (auto&& to : tos) {
			Connect(initial, to + lhsSize, Epsilon);
			Disconnect(rhs.initial + lhsSize, to + lhsSize, Epsilon);
		}
	}

	determined = false;
	isAlternative = true;
	return *this;
}

Fsm& Fsm::operator &= (const Fsm& rhs)
{
	Fsm rhs2(rhs);
	Complement();
	rhs2.Complement();
	*this |= rhs2;
	Complement();
	return *this;
}

Fsm& Fsm::Iterate()
{
	PIRE_IFDEBUG(Cdbg << "Iterating:" << Endl << *this << Endl);
	Resize(Size() + 2);

	Connect(Size() - 2, Size() - 1);
	Connect(Size() - 2, initial);
	ConnectFinal(initial);
	ConnectFinal(Size() - 1);

	ClearFinal();
	SetFinal(Size() - 1, true);
	initial = Size() - 2;

	determined = false;

	PIRE_IFDEBUG(Cdbg << "Iterated:" << Endl << *this << Endl);
	return *this;
}

Fsm& Fsm::Complement()
{
	if (!Determine())
		throw Error("Regexp pattern too complicated");
	Minimize();
	Resize(Size() + 1);
	for (size_t i = 0; i < Size(); ++i)
		if (!IsFinal(i))
			Connect(i, Size() - 1);
	ClearFinal();
	SetFinal(Size() - 1, true);
	determined = false;

	return *this;
}

Fsm Fsm::operator *(size_t count) const
{
	Fsm ret;
	while (count--)
		ret += *this;
	return ret;
}

void Fsm::MakePrefix()
{
	RemoveDeadEnds();
	for (size_t i = 0; i < Size(); ++i)
		if (!m_transitions[i].empty())
			m_final.insert(i);
	ClearHints();
}

void Fsm::MakeSuffix()
{
	for (size_t i = 0; i < Size(); ++i)
		if (i != initial)
			Connect(initial, i);
	ClearHints();
}

Fsm& Fsm::Reverse()
{
	Fsm out;
	out.Resize(Size() + 1);
	out.letters = Letters();

	// Invert transitions
	for (size_t from = 0; from < Size(); ++from)
		for (auto&& i : m_transitions[from])
			for (auto&& j : i.second)
				out.Connect(j, from, i.first);

	// Invert initial and final states
	out.m_final.clear();
	out.SetFinal(initial, true);
	for (auto i : m_final)
		out.Connect(Size(), i, Epsilon);
	out.SetInitial(Size());

	// Invert outputs
	for (auto&& i : outputs)
		for (auto&& j : i.second)
			out.SetOutput(j.first, i.first, j.second);

	// Preserve tags (although thier semantics are usually heavily broken at this point)
	out.tags = tags;

	// Apply
	Swap(out);
	return *this;
}

TSet<size_t> Fsm::DeadStates() const
{
	TSet<size_t> res;

	for (int invert = 0; invert <= 1; ++invert) {
		Fsm digraph;
		digraph.Resize(Size());
		for (TransitionTable::const_iterator j = m_transitions.begin(), je = m_transitions.end(); j != je; ++j) {
			for (TransitionRow::const_iterator k = j->begin(), ke = j->end(); k != ke; ++k) {
				for (StatesSet::const_iterator toSt = k->second.begin(), toSte = k->second.end(); toSt != toSte; ++toSt) {
					// We only care if the states are connected or not regerdless through what letter
					if (invert) {
						// Build an FSM with inverted transitions
						digraph.Connect(*toSt, j - m_transitions.begin(), 0);
					} else {
						digraph.Connect(j - m_transitions.begin(), *toSt, 0);
					}
				}
			}
		}

		TVector<bool> unchecked(Size(), true);
		TVector<bool> useless(Size(), true);
		TDeque<size_t> queue;

		// Put all final (or initial) states into queue, marking them useful
		for (size_t i = 0; i < Size(); ++i)
			if ((invert && IsFinal(i)) || (!invert && Initial() == i)) {
				useless[i] = false;
				queue.push_back(i);
			}

		// Do the breadth-first search, marking all states
		// from which already marked states are reachable
		while (!queue.empty()) {
			size_t to = queue.front();
			queue.pop_front();

			// All the states that are connected to this state in the transition matrix are useful
			const StatesSet& connections = (digraph.m_transitions[to])[0];
			for (auto&& fr : connections) {
				// Enqueue the state for further traversal if it hasnt been already checked
				if (unchecked[fr] && useless[fr]) {
					useless[fr] = false;
					queue.push_back(fr);
				}
			}

			// Now we consider this state checked
			unchecked[to] = false;
		}

		for (size_t i = 0; i < Size(); ++i) {
			if (useless[i]) {
				res.insert(i);
			}
		}
	}

	return res;
}

void Fsm::RemoveDeadEnds()
{
	PIRE_IFDEBUG(Cdbg << "Removing dead ends on:" << Endl << *this << Endl);

	TSet<size_t> dead = DeadStates();
	// Erase all useless states
	for (auto&& i : dead) {
		PIRE_IFDEBUG(Cdbg << "Removing useless state " << i << Endl);
		m_transitions[i].clear();
		for (auto&& j : m_transitions)
			for (auto&& k : j)
				k.second.erase(i);
	}
	ClearHints();

	PIRE_IFDEBUG(Cdbg << "Result:" << Endl << *this << Endl);
}

// This method is one step of Epsilon-connection removal algorithm.
// It merges transitions, tags, and outputs of 'to' state into 'from' state
void Fsm::MergeEpsilonConnection(size_t from, size_t to)
{
	unsigned long frEpsOutput = 0;
	bool fsEpsOutputExists = false;

	// Is there an output for 'from'->'to' transition?
	if (outputs.find(from) != outputs.end() && outputs[from].find(to) != outputs[from].end()) {
		frEpsOutput = outputs[from][to];
		fsEpsOutputExists = true;
	}

	// Merge transitions from 'to' state into transitions from 'from' state
	for (auto&& transition : m_transitions[to]) {
		TSet<size_t> connStates;
		std::copy(transition.second.begin(), transition.second.end(),
			std::inserter(m_transitions[from][transition.first], m_transitions[from][transition.first].end()));

		// If there is an output of the 'from'->'to' connection it has to be set to all
		// new connections that were merged from 'to' state
		if (fsEpsOutputExists) {
			// Compute the set of states that are reachable from 'to' state
			std::copy(transition.second.begin(), transition.second.end(), std::inserter(connStates, connStates.end()));

			// For each of these states add an output equal to the Epsilon-connection output
			for (auto&& newConnSt : connStates) {
				outputs[from][newConnSt] |= frEpsOutput;
			}
		}
	}

	// Mark 'from' state final if 'to' state is final
	if (IsFinal(to))
		SetFinal(from, true);

	// Combine tags
	auto ti = tags.find(to);
	if (ti != tags.end())
		tags[from] |= ti->second;

	// Merge all 'to' into 'from' outputs:
	//      outputs[from][i] |= (outputs[from][to] | outputs[to][i])
	auto toOit = outputs.find(to);
	if (toOit != outputs.end()) {
		for (auto&& output : toOit->second) {
			outputs[from][output.first] |= (frEpsOutput | output.second);
		}
	}
}

// Assuming the epsilon transitions is possible from 'from' to 'thru',
// finds all states which are Epsilon-reachable from 'thru' and connects
// them directly to 'from' with Epsilon transition having proper output.
// Updates inverse map of epsilon transitions as well.
void Fsm::ShortCutEpsilon(size_t from, size_t thru, TVector< TSet<size_t> >& inveps)
{
	PIRE_IFDEBUG(Cdbg << "In Fsm::ShortCutEpsilon(" << from << ", " << thru << ")\n");
	const StatesSet& to = Destinations(thru, Epsilon);
	Outputs::iterator outIt = outputs.find(from);
	unsigned long fromThruOut = Output(from, thru);
	for (auto&& toElement : to) {
		PIRE_IFDEBUG(Cdbg << "Epsilon connecting " << from << " --> " << thru << " --> " << toElement << "\n");
		Connect(from, toElement, Epsilon);
		inveps[toElement].insert(from);
		if (outIt != outputs.end())
			outIt->second[toElement] |= (fromThruOut | Output(thru, toElement));
	}
}

// Removes all Epsilon-connections by iterating though states and merging each Epsilon-connection
// effects from 'to' state into 'from' state
void Fsm::RemoveEpsilons()
{
	Unsparse();

	// Build inverse map of epsilon transitions
	TVector< TSet<size_t> > inveps(Size()); // We have to use TSet<> here since we want it sorted
	for (size_t from = 0; from != Size(); ++from) {
		const StatesSet& tos = Destinations(from, Epsilon);
		for (auto&& to : tos)
			inveps[to].insert(from);
	}

	// Make a transitive closure of all epsilon transitions (Floyd-Warshall algorithm)
	// (if there exists an epsilon-path between two states, epsilon-connect them directly)
	for (size_t thru = 0; thru != Size(); ++thru)
		for (auto&& from : inveps[thru])
			// inveps[thru] may alter during loop body, hence we cannot cache ivneps[thru].end()
			if (from != thru)
				ShortCutEpsilon(from, thru, inveps);

	PIRE_IFDEBUG(Cdbg << "=== After epsilons shortcut\n" << *this << Endl);

	// Iterate through all epsilon-connected state pairs, merging states together
	for (size_t from = 0; from != Size(); ++from) {
		const StatesSet& to = Destinations(from, Epsilon);
		for (auto&& toElement : to)
			if (toElement != from)
				MergeEpsilonConnection(from, toElement); // it's a NOP if to == from, so don't waste time
	}

	PIRE_IFDEBUG(Cdbg << "=== After epsilons merged\n" << *this << Endl);

	// Drop all epsilon transitions
	for (auto&& i : m_transitions)
		i.erase(Epsilon);

	Sparse();
	ClearHints();
}

bool Fsm::LettersEquality::operator()(Char a, Char b) const
{
	for (auto&& outer : *m_tbl) {
		auto ia = outer.find(a);
		auto ib = outer.find(b);
		if (ia == outer.end() && ib == outer.end())
			continue;
		else if (ia == outer.end() || ib == outer.end() || ia->second != ib->second) {
			return false;
		}
	}
	return true;
}

void Fsm::Sparse(bool needEpsilons /* = false */)
{
	letters = LettersTbl(LettersEquality(m_transitions));
	for (unsigned letter = 0; letter < MaxChar; ++letter)
		if (letter != Epsilon || needEpsilons)
			letters.Append(letter);

	m_sparsed = true;
	PIRE_IFDEBUG(Cdbg << "Letter classes = " << letters << Endl);
}

void Fsm::Unsparse()
{
	for (auto&& letter : letters)
		for (auto&& i : m_transitions)
			for (auto&& j : letter.second.second)
				i[j] = i[letter.first];
	m_sparsed = false;
}

// Returns a set of 'terminal states', which are those of the final states,
// from which a transition to themselves on any letter is possible.
TSet<size_t> Fsm::TerminalStates() const
{
	TSet<size_t> terminals;
	for (auto&& final : m_final) {
		bool ok = true;
		for (auto&& letter : letters) {
			auto dests = m_transitions[final].find(letter.first);
			ok = ok && (dests != m_transitions[final].end() && dests->second.find(final) != dests->second.end());
		}
		if (ok)
			terminals.insert(final);
	}
	return terminals;
}

namespace Impl {
class FsmDetermineTask {
public:
	typedef TVector<size_t> State;
	typedef Fsm::LettersTbl LettersTbl;
	typedef TMap<State, size_t> InvStates;

	FsmDetermineTask(const Fsm& fsm)
		: mFsm(fsm)
		, mTerminals(fsm.TerminalStates())
	{
		PIRE_IFDEBUG(Cdbg << "Terminal states: [" << Join(mTerminals.begin(), mTerminals.end(), ", ") << "]" << Endl);
	}
	const LettersTbl& Letters() const { return mFsm.letters; }

	State Initial() const { return State(1, mFsm.initial); }
	bool IsRequired(const State& state) const
	{
		for (auto&& i : state)
			if (mTerminals.find(i) != mTerminals.end())
				return false;
		return true;
	}

	State Next(const State& state, Char letter) const
	{
		State next;
		next.reserve(20);
		for (auto&& from : state) {
			const auto& part = mFsm.Destinations(from, letter);
			std::copy(part.begin(), part.end(), std::back_inserter(next));
		}

		std::sort(next.begin(), next.end());
		next.erase(std::unique(next.begin(), next.end()), next.end());
		PIRE_IFDEBUG(Cdbg << "Returning transition [" << Join(state.begin(), state.end(), ", ") << "] --" << letter
		                  << "--> [" << Join(next.begin(), next.end(), ", ") << "]" << Endl);
		return next;
	}

	void AcceptStates(const TVector<State>& states)
	{
		mNewFsm.Resize(states.size());
		mNewFsm.initial = 0;
		mNewFsm.determined = true;
		mNewFsm.letters = Letters();
		mNewFsm.m_final.clear();
		for (size_t ns = 0; ns < states.size(); ++ns) {
			PIRE_IFDEBUG(Cdbg << "State " << ns << " = [" << Join(states[ns].begin(), states[ns].end(), ", ") << "]" << Endl);
			for (auto&& j : states[ns]) {

				// If it was a terminal state, connect it to itself
				if (mTerminals.find(j) != mTerminals.end()) {
					for (auto&& letter : Letters())
						mNewFsm.Connect(ns, ns, letter.first);
					mNewTerminals.insert(ns);
					PIRE_IFDEBUG(Cdbg << "State " << ns << " becomes terminal because of old state " << j << Endl);
				}
			}
			for (auto&& j : states[ns]) {
				// If any state containing in our one is marked final, mark the new state final as well
				if (mFsm.IsFinal(j)) {
					PIRE_IFDEBUG(Cdbg << "State " << ns << " becomes final because of old state " << j << Endl);
					mNewFsm.SetFinal(ns, true);
					if (mFsm.tags.empty())
						// Weve got no tags and already know that the state is final,
						// hence weve done with this state and got nothing more to do.
						break;
				}

				// Bitwise OR all tags in states
				auto ti = mFsm.tags.find(j);
				if (ti != mFsm.tags.end()) {
					PIRE_IFDEBUG(Cdbg << "State " << ns << " carries tag " << ti->second << " because of old state " << j << Endl);
					mNewFsm.tags[ns] |= ti->second;
				}
			}
		}
		// For each old state, prepare a list of new state it is contained in
		typedef TMap< size_t, TVector<size_t> > Old2New;
		Old2New old2new;
		for (size_t ns = 0; ns < states.size(); ++ns)
			for (auto&& j : states[ns])
				old2new[j].push_back(ns);

		// Copy all outputs
		for (auto&& i : mFsm.outputs) {
			for (auto&& j : i.second) {
				auto from = old2new.find(i.first);
				auto to = old2new.find(j.first);
				if (from != old2new.end() && to != old2new.end()) {
					for (auto&& k : from->second)
						for (auto&& l : to->second)
							mNewFsm.outputs[k][l] |= j.second;
				}
			}
		}
		PIRE_IFDEBUG(Cdbg << "New terminals = [" << Join(mNewTerminals.begin(), mNewTerminals.end(), ",") << "]" << Endl);
	}

	void Connect(size_t from, size_t to, Char letter)
	{
		PIRE_IFDEBUG(Cdbg << "Connecting " << from << " --" << letter << "--> " << to << Endl);
		Y_ASSERT(mNewTerminals.find(from) == mNewTerminals.end());
		mNewFsm.Connect(from, to, letter);
	}
	typedef bool Result;

	Result Success() {
		Fsm::Outputs oldOutputs;
		// remove redundant outputs
		oldOutputs.swap(mNewFsm.outputs);
		for (size_t from = 0; from < mNewFsm.Size(); ++from) {
			auto fromOutput = oldOutputs.find(from);
			if (fromOutput == oldOutputs.end())
				continue;
			const auto& newTransitionsRow = mNewFsm.m_transitions[from];
			for (auto&& row : newTransitionsRow) {
				for (auto&& stateIt : row.second) {
					auto toOutput = fromOutput->second.find(stateIt);
					if (toOutput != fromOutput->second.end()) {
						mNewFsm.outputs[from].insert(*toOutput);
					}
				}
			}
		}
		return true;
	}

	Result Failure() { return false; }

	Fsm& Output() { return mNewFsm; }
private:
	const Fsm& mFsm;
	Fsm mNewFsm;
	TSet<size_t> mTerminals;
	TSet<size_t> mNewTerminals;
};
}

bool Fsm::Determine(size_t maxsize /* = 0 */)
{
	static const unsigned MaxSize = 200000;
	if (determined)
		return true;

	PIRE_IFDEBUG(Cdbg << "=== Initial ===" << Endl << *this << Endl);

	RemoveEpsilons();
	PIRE_IFDEBUG(Cdbg << "=== After all epsilons removed" << Endl << *this << Endl);

	Impl::FsmDetermineTask task(*this);
	if (Pire::Impl::Determine(task, maxsize ? maxsize : MaxSize)) {
		task.Output().Swap(*this);
		PIRE_IFDEBUG(Cdbg << "=== Determined ===" << Endl << *this << Endl);
		return true;
	} else
		return false;
}

namespace Impl {
class FsmMinimizeTask {
public:
	explicit FsmMinimizeTask(const Fsm& fsm)
		: mFsm(fsm)
		, reversedTransitions(fsm.Size())
		, StateClass(fsm.Size())
		, Classes(0)
	{
		Y_ASSERT(mFsm.IsDetermined());

		TMap<bool, size_t> FinalStateClassMap;

		for (size_t state = 0; state < mFsm.Size(); ++state) {
			reversedTransitions[state].resize(mFsm.Letters().Size());
			if (FinalStateClassMap.find(mFsm.IsFinal(state)) == FinalStateClassMap.end()) {
				FinalStateClassMap[mFsm.IsFinal(state)] = Classes++;
			}
			StateClass[state] = FinalStateClassMap[mFsm.IsFinal(state)];
		}

		for (size_t state = 0; state < mFsm.Size(); ++state) {
			TSet<ypair<Char, size_t>> usedTransitions;
			for (const auto& transition : mFsm.m_transitions[state]) {
				Y_ASSERT(transition.second.size() == 1);
				auto destination = *transition.second.begin();
				auto letter = mFsm.Letters().Index(transition.first);
				if (usedTransitions.find(ymake_pair(letter, destination)) == usedTransitions.end()) {
					usedTransitions.insert(ymake_pair(letter, destination));
					reversedTransitions[destination][letter].push_back(state);
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
		return mFsm.IsDetermined();
	}

	size_t Size() const {
		return mFsm.Size();
	}

	const TVector<size_t>& Previous(size_t state, size_t letter) const {
		return reversedTransitions[state][letter];
	}

	void AcceptStates() {
		mNewFsm.Resize(Classes);
		mNewFsm.letters = mFsm.letters;
		mNewFsm.determined = mFsm.determined;
		mNewFsm.m_sparsed = mFsm.m_sparsed;
		mNewFsm.SetFinal(0, false);

		// Unite equality classes into new states
		size_t fromIdx = 0;
		for (auto from = mFsm.m_transitions.begin(), fromEnd = mFsm.m_transitions.end(); from != fromEnd; ++from, ++fromIdx) {
			size_t dest = StateClass[fromIdx];
			PIRE_IFDEBUG(Cdbg << "[min] State " << fromIdx << " becomes state " << dest << Endl);
			for (auto&& letter : *from) {
				Y_ASSERT(letter.second.size() == 1 || !"FSM::minimize(): FSM not deterministic");
				mNewFsm.Connect(dest, StateClass[*letter.second.begin()], letter.first);
			}
			if (mFsm.IsFinal(fromIdx)) {
				mNewFsm.SetFinal(dest, true);
				PIRE_IFDEBUG(Cdbg << "[min] New state " << dest << " becomes final because of old state " << fromIdx << Endl);
			}

			// Append tags
			auto ti = mFsm.tags.find(fromIdx);
			if (ti != mFsm.tags.end()) {
				mNewFsm.tags[dest] |= ti->second;
				PIRE_IFDEBUG(Cdbg << "[min] New state " << dest << " carries tag " << ti->second << " because of old state " << fromIdx << Endl);
			}
		}
		mNewFsm.initial = StateClass[mFsm.initial];

		// Restore outputs
		for (auto&& output : mFsm.outputs)
			for (auto&& output2 : output.second)
				mNewFsm.outputs[StateClass[output.first]].insert(ymake_pair(StateClass[output2.first], output2.second));
	}

	typedef bool Result;

	Result Success() {
		return true;
	}

	Result Failure() {
		return false;
	}

	Fsm& Output() {
		return mNewFsm;
	}

private:
	const Fsm& mFsm;
	Fsm mNewFsm;
	TVector<TVector<TVector<size_t>>> reversedTransitions;
	TVector<size_t> StateClass;
	size_t Classes;
};
}

void Fsm::Minimize()
{
	// Minimization algorithm is only applicable to a determined FSM.
	Y_ASSERT(determined);

	Impl::FsmMinimizeTask task{*this};
	if (Pire::Impl::Minimize(task)) {
		task.Output().Swap(*this);
	}
}

Fsm& Fsm::Canonize(size_t maxSize /* = 0 */)
{
	if (!IsDetermined()) {
		if (!Determine(maxSize))
			throw Error("regexp pattern too complicated");
	}
	Minimize();
	return *this;
}

void Fsm::PrependAnything()
{
	size_t newstate = Size();
	Resize(Size() + 1);
	for (size_t letter = 0; letter < MaxChar; ++letter)
		Connect(newstate, newstate, letter);

	Connect(newstate, initial);
	initial = newstate;

	determined = false;
}

void Fsm::AppendAnything()
{
	size_t newstate = Size();
	Resize(Size() + 1);
	for (size_t letter = 0; letter < MaxChar; ++letter)
		Connect(newstate, newstate, letter);

	ConnectFinal(newstate);
	ClearFinal();
	SetFinal(newstate, 1);

	determined = false;
}

Fsm& Fsm::Surround()
{
	PrependAnything();
	AppendAnything();
	return *this;
}

void Fsm::Divert(size_t from, size_t to, size_t dest)
{
	if (to == dest)
		return;

	// Assign the output
	auto oi = outputs.find(from);
	if (oi != outputs.end()) {
		auto oi2 = oi->second.find(to);
		if (oi2 != oi->second.end()) {
			unsigned long output = oi2->second;
			oi->second.erase(oi2);
			oi->second.insert(ymake_pair(dest, output));
		}
	}

	// Assign the transition
	for (auto&& i : m_transitions[from]) {
		auto di = i.second.find(to);
		if (di != i.second.end()) {
			i.second.erase(di);
			i.second.insert(dest);
		}
	}

	ClearHints();
}


}
