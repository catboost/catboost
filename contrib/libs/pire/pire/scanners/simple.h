/*
 * simple.h -- the definition of the SimpleScanner
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


#ifndef PIRE_SCANNERS_SIMPLE_H
#define PIRE_SCANNERS_SIMPLE_H

#include <contrib/libs/pire/pire/approx_matching.h>
#include <contrib/libs/pire/pire/stub/stl.h>
#include <contrib/libs/pire/pire/stub/defaults.h>
#include <contrib/libs/pire/pire/stub/saveload.h>

#include "common.h"

namespace Pire {

/**
 * More faster version than the Scanner, but incapable of storing multiple
 * regexps and taking more memory for the same regexp.
 */
class SimpleScanner {
private:
	static const size_t STATE_ROW_SIZE = MaxChar + 1; // All characters + 1 element to store final state flag

public:
	typedef size_t      Transition;
	typedef ui16        Letter;
	typedef ui32        Action;
	typedef ui8         Tag;

	SimpleScanner()	{ Alias(Null()); }

	explicit SimpleScanner(Fsm& fsm, size_t distance = 0);

	size_t Size() const { return m.statesCount; }
	bool Empty() const { return m_transitions == Null().m_transitions; }

	typedef size_t State;

	size_t RegexpsCount() const { return Empty() ? 0 : 1; }
	size_t LettersCount() const { return MaxChar; }

	/// Checks whether specified state is in any of the final sets
	bool Final(const State& state) const { return *(((const Transition*) state) - 1) != 0; }

	bool Dead(const State&) const { return false; }

	ypair<const size_t*, const size_t*> AcceptedRegexps(const State& s) const {
		return Final(s) ? Accept() : Deny();
	}

	/// returns an initial state for this scanner
	void Initialize(State& state) const { state = m.initial; }

	/// Handles one characters
	Action Next(State& state, Char c) const
	{
		Transition shift = reinterpret_cast<const Transition*>(state)[c];
		state += shift;
		return 0;
	}

	bool TakeAction(State&, Action) const { return false; }

	SimpleScanner(const SimpleScanner& s): m(s.m)
	{
		if (!s.m_buffer) {
			// Empty or mmap()-ed scanner, just copy pointers
			m_buffer = 0;
			m_transitions = s.m_transitions;
		} else {
			// In-memory scanner, perform deep copy
			m_buffer = BufferType(new char[BufSize()]);
			memcpy(m_buffer.Get(), s.m_buffer.Get(), BufSize());
			Markup(m_buffer.Get());

			m.initial += (m_transitions - s.m_transitions) * sizeof(Transition);
		}
	}

	// Makes a shallow ("weak") copy of the given scanner.
	// The copied scanner does not maintain lifetime of the original's entrails.
	void Alias(const SimpleScanner& s)
	{
		m = s.m;
		m_buffer.Reset();
		m_transitions = s.m_transitions;
	}

	void Swap(SimpleScanner& s)
	{
		DoSwap(m_buffer, s.m_buffer);
		DoSwap(m.statesCount, s.m.statesCount);
		DoSwap(m.initial, s.m.initial);
		DoSwap(m_transitions, s.m_transitions);
	}

	SimpleScanner& operator = (const SimpleScanner& s) { SimpleScanner(s).Swap(*this); return *this; }

	~SimpleScanner() = default;

	/*
	 * Constructs the scanner from mmap()-ed memory range, returning a pointer
	 * to unconsumed part of the buffer.
	 */
	const void* Mmap(const void* ptr, size_t size)
	{
		Impl::CheckAlign(ptr);
		SimpleScanner s;

		const size_t* p = reinterpret_cast<const size_t*>(ptr);
		Impl::ValidateHeader(p, size, ScannerIOTypes::SimpleScanner, sizeof(m));
		if (size < sizeof(s.m))
			throw Error("EOF reached while mapping NPire::Scanner");

		memcpy(&s.m, p, sizeof(s.m));
		Impl::AdvancePtr(p, size, sizeof(s.m));
		Impl::AlignPtr(p, size);

		bool empty = *((const bool*) p);
		Impl::AdvancePtr(p, size, sizeof(empty));
		Impl::AlignPtr(p, size);

		if (empty)
			s.Alias(Null());
		else {
			if (size < s.BufSize())
				throw Error("EOF reached while mapping NPire::Scanner");
			s.Markup(const_cast<size_t*>(p));
			s.m.initial += reinterpret_cast<size_t>(s.m_transitions);

			Swap(s);
			Impl::AdvancePtr(p, size, BufSize());
		}
		return Impl::AlignPtr(p, size);
	}

	size_t StateIndex(State s) const
	{
		return (s - reinterpret_cast<size_t>(m_transitions)) / (STATE_ROW_SIZE * sizeof(Transition));
	}

	// Returns the size of the memory buffer used (or required) by scanner.
	size_t BufSize() const
	{
		return STATE_ROW_SIZE * m.statesCount * sizeof(Transition); // Transitions table
	}

	void Save(yostream*) const;
	void Load(yistream*);

protected:
	struct Locals {
		size_t statesCount;
		size_t initial;
	} m;

	using BufferType = TArrayHolder<char>;
	BufferType m_buffer;

	Transition* m_transitions;

	inline static const SimpleScanner& Null()
	{
		static const SimpleScanner n = Fsm::MakeFalse().Compile<SimpleScanner>();
		return n;
	}

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

	/*
	 * Initializes pointers depending on buffer start, letters and states count
	 */
	void Markup(void* ptr)
	{
		m_transitions = reinterpret_cast<Transition*>(ptr);
	}

	void SetJump(size_t oldState, Char c, size_t newState)
	{
		Y_ASSERT(m_buffer);
		Y_ASSERT(oldState < m.statesCount);
		Y_ASSERT(newState < m.statesCount);
		m_transitions[oldState * STATE_ROW_SIZE + 1 + c]
			= (((newState - oldState) * STATE_ROW_SIZE) * sizeof(Transition));
	}

	unsigned long RemapAction(unsigned long action) { return action; }

	void SetInitial(size_t state)
	{
		Y_ASSERT(m_buffer);
		m.initial = reinterpret_cast<size_t>(m_transitions + state * STATE_ROW_SIZE + 1);
	}

	void SetTag(size_t state, size_t tag)
	{
		Y_ASSERT(m_buffer);
		m_transitions[state * STATE_ROW_SIZE] = tag;
	}

};
inline SimpleScanner::SimpleScanner(Fsm& fsm, size_t distance)
{
	if (distance) {
		fsm = CreateApproxFsm(fsm, distance);
	}
	fsm.Canonize();

	m.statesCount = fsm.Size();
	m_buffer = BufferType(new char[BufSize()]);
	memset(m_buffer.Get(), 0, BufSize());
	Markup(m_buffer.Get());
	m.initial = reinterpret_cast<size_t>(m_transitions + fsm.Initial() * STATE_ROW_SIZE + 1);
	for (size_t state = 0; state < fsm.Size(); ++state)
		SetTag(state, fsm.Tag(state) | (fsm.IsFinal(state) ? 1 : 0));

	for (size_t from = 0; from != fsm.Size(); ++from)
		for (auto&& i : fsm.Letters()) {
			const auto& tos = fsm.Destinations(from, i.first);
			if (tos.empty())
				continue;
			for (auto&& l : i.second.second)
				for (auto&& to : tos)
					SetJump(from, l, to);
		}
}


}

#endif
