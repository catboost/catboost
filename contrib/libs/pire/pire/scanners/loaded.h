/*
 * loaded.h -- a definition of the LoadedScanner
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


#ifndef PIRE_SCANNERS_LOADED_H
#define PIRE_SCANNERS_LOADED_H

#include <string.h>

#include <contrib/libs/pire/pire/approx_matching.h>
#include <contrib/libs/pire/pire/fsm.h>
#include <contrib/libs/pire/pire/partition.h>

#include "common.h"

#ifdef PIRE_DEBUG
#include <iostream>
#endif

namespace Pire {

/**
* A loaded scanner -- the deterministic scanner having actions
*                     associated with states and transitions
*
* Not a complete scanner itself (hence abstract), this class provides
* infrastructure for regexp-based algorithms (e.g. counts or captures),
* supporting major part of scanner construction, (de)serialization,
* mmap()-ing, etc.
*
* It is a good idea to override copy ctor, operator= and swap()
* in subclasses to avoid mixing different scanner types in these methods.
* Also please note that subclasses should not have any data members of thier own.
*/
class LoadedScanner {
public:
	typedef ui8         Letter;
	typedef ui32        Action;
	typedef ui8         Tag;

	typedef size_t InternalState;

	union Transition {
		size_t raw;			// alignment hint for compiler
		struct {
			ui32 shift;
			Action action;
		};
	};

	// Override in subclass, if neccessary
	enum {
		FinalFlag = 0,
		DeadFlag  = 0
	};

	static const size_t MAX_RE_COUNT = 16;

protected:
	LoadedScanner() { Alias(Null()); }

	LoadedScanner(const LoadedScanner& s): m(s.m)
	{
		if (s.m_buffer) {
			m_buffer = BufferType(new char [BufSize()]);
			memcpy(m_buffer.Get(), s.m_buffer.Get(), BufSize());
			Markup(m_buffer.Get());
			m.initial = (InternalState)m_jumps + (s.m.initial - (InternalState)s.m_jumps);
		} else {
			Alias(s);
		}
	}

	void Swap(LoadedScanner& s)
	{
		DoSwap(m_buffer, s.m_buffer);
		DoSwap(m.statesCount, s.m.statesCount);
		DoSwap(m.lettersCount, s.m.lettersCount);
		DoSwap(m.regexpsCount, s.m.regexpsCount);
		DoSwap(m.initial, s.m.initial);
		DoSwap(m_letters, s.m_letters);
		DoSwap(m_jumps, s.m_jumps);
		DoSwap(m_tags, s.m_tags);
	}

	LoadedScanner& operator = (const LoadedScanner& s) { LoadedScanner(s).Swap(*this); return *this; }
	LoadedScanner (LoadedScanner&& other) : LoadedScanner() {
		Swap(other);
	}
	LoadedScanner& operator=(LoadedScanner&& other) {
		Swap(other);
		return *this;
	}

public:
	size_t Size() const { return m.statesCount; }

	bool Empty() const { return m_jumps == Null().m_jumps; }

	size_t RegexpsCount() const { return Empty() ? 0 : m.regexpsCount; }

	size_t LettersCount() const { return m.lettersCount; }

	const void* Mmap(const void* ptr, size_t size) {
		return Mmap(ptr, size, nullptr);
	}

	const void* Mmap(const void* ptr, size_t size, ui32* type)
	{
		Impl::CheckAlign(ptr);
		LoadedScanner s;
		const size_t* p = reinterpret_cast<const size_t*>(ptr);
		Header header = Impl::ValidateHeader(p, size, ScannerIOTypes::LoadedScanner, sizeof(s.m));
		if (type) {
			*type = header.Type;
		}

		Locals* locals;
		Impl::MapPtr(locals, 1, p, size);
		memcpy(&s.m, locals, sizeof(s.m));

		Impl::MapPtr(s.m_letters, MaxChar, p, size);
		Impl::MapPtr(s.m_jumps, s.m.statesCount * s.m.lettersCount, p, size);
		if (header.Version == Header::RE_VERSION_WITH_MACTIONS) {
			Action* actions = 0;
			Impl::MapPtr(actions, s.m.statesCount * s.m.lettersCount, p, size);
		}
		Impl::MapPtr(s.m_tags, s.m.statesCount, p, size);

		s.m.initial += reinterpret_cast<size_t>(s.m_jumps);
		Swap(s);

		return (const void*) p;
	}

	void Save(yostream*, ui32 type) const;
	void Save(yostream*) const;
	void Load(yistream*, ui32* type);
	void Load(yistream*);

		template<class Eq>
	void Init(size_t states, const Partition<Char, Eq>& letters, size_t startState, size_t regexpsCount = 1)
	{
		m.statesCount = states;
		m.lettersCount = letters.Size();
		m.regexpsCount = regexpsCount;
		m_buffer = BufferType(new char[BufSize()]);
		memset(m_buffer.Get(), 0, BufSize());
		Markup(m_buffer.Get());

		m.initial = reinterpret_cast<size_t>(m_jumps + startState * m.lettersCount);

		// Build letter translation table
		Fill(m_letters, m_letters + MaxChar, 0);
		for (auto&& letter : letters)
			for (auto&& character : letter.second.second)
				m_letters[character] = letter.second.first;
	}

	size_t StateSize() const
	{
		return m.lettersCount * sizeof(*m_jumps);
	}

	size_t TransitionIndex(size_t state, Char c) const
	{
		return state * m.lettersCount + m_letters[c];
	}

	void SetJump(size_t oldState, Char c, size_t newState, Action action)
	{
		Y_ASSERT(m_buffer);
		Y_ASSERT(oldState < m.statesCount);
		Y_ASSERT(newState < m.statesCount);

		size_t shift = (newState - oldState) * StateSize();
		Transition tr;
		tr.shift = (ui32)shift;
		tr.action = action;
		m_jumps[TransitionIndex(oldState, c)] = tr;
	}

	Action RemapAction(Action action) { return action; }

	void SetInitial(size_t state) { Y_ASSERT(m_buffer); m.initial = reinterpret_cast<size_t>(m_jumps + state * m.lettersCount); }
	void SetTag(size_t state, Tag tag) { Y_ASSERT(m_buffer); m_tags[state] = tag; }
	void FinishBuild() {}

	size_t StateIdx(InternalState s) const
	{
		return (reinterpret_cast<Transition*>(s) - m_jumps) / m.lettersCount;
	}

	i64 SignExtend(i32 i) const { return i; }

	size_t BufSize() const
	{
		return
			MaxChar * sizeof(*m_letters)
			+ m.statesCount * StateSize()
			+ m.statesCount * sizeof(*m_tags)
			;
	}

protected:

	static const Action IncrementMask     = (1 << MAX_RE_COUNT) - 1;
	static const Action ResetMask         = IncrementMask << MAX_RE_COUNT;

	// TODO: maybe, put fields in private section and provide data accessors

	struct Locals {
		ui32 statesCount;
		ui32 lettersCount;
		ui32 regexpsCount;
		size_t initial;
	} m;

	using BufferType = TArrayHolder<char>;
	BufferType m_buffer;

	Letter* m_letters;
	Transition* m_jumps;
	Tag* m_tags;

	virtual ~LoadedScanner();

private:
	explicit LoadedScanner(Fsm& fsm, size_t distance = 0)
	{
		if (distance) {
			fsm = CreateApproxFsm(fsm, distance);
		}
		fsm.Canonize();
		Init(fsm.Size(), fsm.Letters(), fsm.Initial());
		BuildScanner(fsm, *this);
	}

	inline static const LoadedScanner& Null()
	{
		static const LoadedScanner n = Fsm::MakeFalse().Compile<LoadedScanner>();
		return n;
	}

	void Markup(void* buf)
	{
		m_letters = reinterpret_cast<Letter*>(buf);
		m_jumps   = reinterpret_cast<Transition*>(m_letters + MaxChar);
		m_tags    = reinterpret_cast<Tag*>(m_jumps + m.statesCount * m.lettersCount);
	}

	void Alias(const LoadedScanner& s)
	{
		memcpy(&m, &s.m, sizeof(m));
		m_buffer = 0;
		m_letters = s.m_letters;
		m_jumps = s.m_jumps;
		m_tags = s.m_tags;
	}

	template<class Eq>
	LoadedScanner(size_t states, const Partition<Char, Eq>& letters, size_t startState, size_t regexpsCount = 1)
	{
		Init(states, letters, startState, regexpsCount);
	}

	friend class Fsm;
};

inline LoadedScanner::~LoadedScanner() = default;

}


#endif
