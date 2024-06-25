/*
 * multi.h -- definition of the Scanner
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


#ifndef PIRE_SCANNERS_MULTI_H
#define PIRE_SCANNERS_MULTI_H

#include <cstring>
#include <string.h>
#include <contrib/libs/pire/pire/approx_matching.h>
#include <contrib/libs/pire/pire/fsm.h>
#include <contrib/libs/pire/pire/partition.h>
#include <contrib/libs/pire/pire/run.h>
#include <contrib/libs/pire/pire/static_assert.h>
#include <contrib/libs/pire/pire/platform.h>
#include <contrib/libs/pire/pire/glue.h>
#include <contrib/libs/pire/pire/determine.h>
#include <contrib/libs/pire/pire/stub/stl.h>
#include <contrib/libs/pire/pire/stub/saveload.h>
#include <contrib/libs/pire/pire/stub/lexical_cast.h>

#include "common.h"

namespace Pire {

namespace Impl {

	inline static ssize_t SignExtend(i32 i) { return i; }
	template<class T>
	class ScannerGlueCommon;

	template<class T>
	class ScannerGlueTask;

	// This strategy allows to mmap() saved representation of a scanner. This is achieved by
	// storing shifts instead of addresses in the transition table.
	struct Relocatable {
		static const size_t Signature = 1;
		// Please note that Transition size is hardcoded as 32 bits.
		// This limits size of transition table to 4G, but compresses
		// it twice compared to 64-bit transitions. In future Transition
		// can be made a template parameter if this is a concern.
		typedef ui32 Transition;

		typedef const void* RetvalForMmap;

		static size_t Go(size_t state, Transition shift) { return state + SignExtend(shift); }
		static Transition Diff(size_t from, size_t to) { return static_cast<Transition>(to - from); }
	};

	// With this strategy the transition table stores addresses. This makes the scanner faster
	// compared to mmap()-ed
	struct Nonrelocatable {
		static const size_t Signature = 2;
		typedef size_t Transition;

		// Generates a compile-time error if Scanner<Nonrelocatable>::Mmap()
		// (which is unsupported) is mistakenly called
		typedef struct {} RetvalForMmap;

		static size_t Go(size_t /*state*/, Transition shift) { return shift; }
		static Transition Diff(size_t /*from*/, size_t to) { return to; }
	};


// Scanner implementation parametrized by
//      - transition table representation strategy
//      - strategy for fast forwarding through memory ranges
template<class Relocation, class Shortcutting>
class Scanner {
protected:
	enum {
		 FinalFlag = 1,
		 DeadFlag  = 2,
		 Flags = FinalFlag | DeadFlag
	};

	static const size_t End = static_cast<size_t>(-1);

public:
	typedef typename Relocation::Transition Transition;

	typedef ui16		Letter;
	typedef ui32		Action;
	typedef ui8		Tag;

	/// Some properties of the particular state.
	struct CommonRowHeader {
		size_t Flags; ///< Holds FinalFlag, DeadFlag, etc...

		CommonRowHeader(): Flags(0) {}

		template <class OtherCommonRowHeader>
		CommonRowHeader& operator =(const OtherCommonRowHeader& other)
		{
			Flags = other.Flags;
			return *this;
		}
	};

	typedef typename Shortcutting::template ExtendedRowHeader<Scanner> ScannerRowHeader;

	Scanner() { Alias(Null()); }

	explicit Scanner(Fsm& fsm, size_t distance = 0)
	{
		if (distance) {
			fsm = CreateApproxFsm(fsm, distance);
		}
		fsm.Canonize();
		Init(fsm.Size(), fsm.Letters(), fsm.Finals().size(), fsm.Initial(), 1);
		BuildScanner(fsm, *this);
	}


	size_t Size() const { return m.statesCount; }
	bool Empty() const { return m_transitions == Null().m_transitions; }

	typedef size_t State;

	size_t RegexpsCount() const { return Empty() ? 0 : m.regexpsCount; }
	size_t LettersCount() const { return m.lettersCount; }

	/// Checks whether specified state is in any of the final sets
	bool Final(const State& state) const { return (Header(state).Common.Flags & FinalFlag) != 0; }

	/// Checks whether specified state is 'dead' (i.e. scanner will never
	/// reach any final state from current one)
	bool Dead(const State& state) const { return (Header(state).Common.Flags & DeadFlag) != 0; }

	ypair<const size_t*, const size_t*> AcceptedRegexps(const State& state) const
	{
		size_t idx = (state - reinterpret_cast<size_t>(m_transitions)) /
			(RowSize() * sizeof(Transition));
		const size_t* b = m_final + m_finalIndex[idx];
		const size_t* e = b;
		while (*e != End)
			++e;
		return ymake_pair(b, e);
	}

	/// Returns an initial state for this scanner
	void Initialize(State& state) const { state = m.initial; }

	Char Translate(Char ch) const
	{
		return m_letters[static_cast<size_t>(ch)];
	}

	/// Handles one letter
	Action NextTranslated(State& state, Char letter) const
	{
		PIRE_IFDEBUG(
			Y_ASSERT(state >= (size_t)m_transitions);
			Y_ASSERT(state < (size_t)(m_transitions + RowSize()*Size()));
			Y_ASSERT((state - (size_t)m_transitions) % (RowSize()*sizeof(Transition)) == 0);
		);

		state = Relocation::Go(state, reinterpret_cast<const Transition*>(state)[letter]);

		PIRE_IFDEBUG(
			Y_ASSERT(state >= (size_t)m_transitions);
			Y_ASSERT(state < (size_t)(m_transitions + RowSize()*Size()));
			Y_ASSERT((state - (size_t)m_transitions) % (RowSize()*sizeof(Transition)) == 0);
		);

		return 0;
	}

	/// Handles one character
	Action Next(State& state, Char c) const
	{
		return NextTranslated(state, Translate(c));
	}

	void TakeAction(State&, Action) const {}

	Scanner(const Scanner& s): m(s.m)
	{
		if (!s.m_buffer) {
			// Empty or mmap()-ed scanner
			Alias(s);
		} else {
			// In-memory scanner
			DeepCopy(s);
		}
	}

	Scanner(Scanner&& s)
	{
		Alias(Null());
		Swap(s);
	}

	template<class AnotherRelocation>
	Scanner(const Scanner<AnotherRelocation, Shortcutting>& s)
	{
		if (s.Empty())
			Alias(Null());
		else
			DeepCopy(s);
	}

	void Swap(Scanner& s)
	{
		Y_ASSERT(m.relocationSignature == s.m.relocationSignature);
		Y_ASSERT(m.shortcuttingSignature == s.m.shortcuttingSignature);
		DoSwap(m_buffer, s.m_buffer);
		DoSwap(m.statesCount, s.m.statesCount);
		DoSwap(m.lettersCount, s.m.lettersCount);
		DoSwap(m.regexpsCount, s.m.regexpsCount);
		DoSwap(m.initial, s.m.initial);
		DoSwap(m_letters, s.m_letters);
		DoSwap(m.finalTableSize, s.m.finalTableSize);
		DoSwap(m_final, s.m_final);
		DoSwap(m_finalIndex, s.m_finalIndex);
		DoSwap(m_transitions, s.m_transitions);
	}

	Scanner& operator = (const Scanner& s) { Scanner(s).Swap(*this); return *this; }

	/*
	 * Constructs the scanner from mmap()-ed memory range, returning a pointer
	 * to unconsumed part of the buffer.
	 */
	typename Relocation::RetvalForMmap Mmap(const void* ptr, size_t size)
	{
		Impl::CheckAlign(ptr, sizeof(size_t));
		Scanner s;

		const size_t* p = reinterpret_cast<const size_t*>(ptr);
		Impl::ValidateHeader(p, size, ScannerIOTypes::Scanner, sizeof(m));
		if (size < sizeof(s.m))
			throw Error("EOF reached while mapping Pire::Scanner");

		memcpy(&s.m, p, sizeof(s.m));
		if (s.m.relocationSignature != Relocation::Signature)
			throw Error("Type mismatch while mmapping Pire::Scanner");
		Impl::AdvancePtr(p, size, sizeof(s.m));
		Impl::AlignPtr(p, size);

		if (Shortcutting::Signature != s.m.shortcuttingSignature)
			throw Error("This scanner has different shortcutting type");

		bool empty = *((const bool*) p);
		Impl::AdvancePtr(p, size, sizeof(empty));
		Impl::AlignPtr(p, size);

		if (empty)
			s.Alias(Null());
		else {
			if (size < s.BufSize())
				throw Error("EOF reached while mapping NPire::Scanner");
			s.Markup(const_cast<size_t*>(p));
			Impl::AdvancePtr(p, size, s.BufSize());
			s.m.initial += reinterpret_cast<size_t>(s.m_transitions);
		}

		Swap(s);
		return Impl::AlignPtr(p, size);
	}

	size_t StateIndex(State s) const
	{
		return (s - reinterpret_cast<size_t>(m_transitions)) / (RowSize() * sizeof(Transition));
	}

	/**
	 * Agglutinates two scanners together, producing a larger scanner.
	 * Checkig a string against that scanner effectively checks them against both agglutinated regexps
	 * (detailed information about matched regexps can be obtained with AcceptedRegexps()).
	 *
	 * Returns default-constructed scanner in case of failure
	 * (consult Scanner::Empty() to find out whether the operation was successful).
	 */
	static Scanner Glue(const Scanner& a, const Scanner& b, size_t maxSize = 0);

	// Returns the size of the memory buffer used (or required) by scanner.
	size_t BufSize() const
	{
		return AlignUp(
			MaxChar * sizeof(Letter)                           // Letters translation table
			+ m.finalTableSize * sizeof(size_t)                // Final table
			+ m.statesCount * sizeof(size_t)                   // Final index
			+ RowSize() * m.statesCount * sizeof(Transition),  // Transitions table
		sizeof(size_t));
	}

	void Save(yostream*) const;
	void Load(yistream*);

	ScannerRowHeader& Header(State s) { return *(ScannerRowHeader*) s; }
	const ScannerRowHeader& Header(State s) const { return *(const ScannerRowHeader*) s; }

protected:

	struct Locals {
		ui32 statesCount;
		ui32 lettersCount;
		ui32 regexpsCount;
		size_t initial;
		ui32 finalTableSize;
		size_t relocationSignature;
		size_t shortcuttingSignature;
	} m;

	using BufferType = TArrayHolder<char>;
	BufferType m_buffer;
	Letter* m_letters;

	size_t* m_final;
	size_t* m_finalIndex;

	Transition* m_transitions;

	inline static const Scanner& Null()
	{
		static const Scanner n = Fsm::MakeFalse().Compile< Scanner<Relocation, Shortcutting> >();

		return n;
	}

	// Returns transition row size in Transition's. Row size_in bytes should be a multiple of sizeof(MaxSizeWord)
	size_t RowSize() const { return AlignUp(m.lettersCount + HEADER_SIZE, sizeof(MaxSizeWord)/sizeof(Transition)); }

	static const size_t HEADER_SIZE = sizeof(ScannerRowHeader) / sizeof(Transition);
	PIRE_STATIC_ASSERT(sizeof(ScannerRowHeader) % sizeof(Transition) == 0);

	template<class Eq>
	void Init(size_t states, const Partition<Char, Eq>& letters, size_t finalStatesCount, size_t startState, size_t regexpsCount = 1)
	{
		std::memset(&m, 0, sizeof(m));
		m.relocationSignature = Relocation::Signature;
		m.shortcuttingSignature = Shortcutting::Signature;
		m.statesCount = states;
		m.lettersCount = letters.Size();
		m.regexpsCount = regexpsCount;
		m.finalTableSize = finalStatesCount + states;

		m_buffer = BufferType(new char[BufSize() + sizeof(size_t)]);
		memset(m_buffer.Get(), 0, BufSize() + sizeof(size_t));
		Markup(AlignUp(m_buffer.Get(), sizeof(size_t)));

		for (size_t i = 0; i != Size(); ++i)
			Header(IndexToState(i)) = ScannerRowHeader();

		m.initial = reinterpret_cast<size_t>(m_transitions + startState * RowSize());

		// Build letter translation table
		for (auto&& letter : letters)
			for (auto&& character : letter.second.second)
				m_letters[character] = letter.second.first + HEADER_SIZE;
	}

	/*
	 * Initializes pointers depending on buffer start, letters and states count
	 */
	void Markup(void* ptr)
	{
		Impl::CheckAlign(ptr, sizeof(size_t));
		m_letters     = reinterpret_cast<Letter*>(ptr);
		m_final	      = reinterpret_cast<size_t*>(m_letters + MaxChar);
		m_finalIndex  = reinterpret_cast<size_t*>(m_final + m.finalTableSize);
		m_transitions = reinterpret_cast<Transition*>(m_finalIndex + m.statesCount);
	}

	// Makes a shallow ("weak") copy of the given scanner.
	// The copied scanner does not maintain lifetime of the original's entrails.
	void Alias(const Scanner<Relocation, Shortcutting>& s)
	{
		memcpy(&m, &s.m, sizeof(m));
		m_buffer.Reset();
		m_letters = s.m_letters;
		m_final = s.m_final;
		m_finalIndex = s.m_finalIndex;
		m_transitions = s.m_transitions;
	}

	template<class AnotherRelocation>
	void DeepCopy(const Scanner<AnotherRelocation, Shortcutting>& s)
	{
		// Don't want memory leaks, but we cannot free the buffer because there might be aliased instances
		Y_ASSERT(m_buffer == nullptr);

		// Ensure that specializations of Scanner across different Relocations do not touch its Locals
		static_assert(sizeof(m) == sizeof(s.m), "sizeof(m) == sizeof(s.m)");
		memcpy(&m, &s.m, sizeof(s.m));
		m.relocationSignature = Relocation::Signature;
		m.shortcuttingSignature = Shortcutting::Signature;
		m_buffer = BufferType(new char[BufSize() + sizeof(size_t)]);
		std::memset(m_buffer.Get(), 0, BufSize() + sizeof(size_t));
		Markup(AlignUp(m_buffer.Get(), sizeof(size_t)));

		// Values in letter-to-leterclass table take into account row header size
		for (size_t c = 0; c < MaxChar; ++c) {
			m_letters[c] = s.m_letters[c] - s.HEADER_SIZE + HEADER_SIZE;
			Y_ASSERT(c == Epsilon || m_letters[c] >= HEADER_SIZE);
			Y_ASSERT(c == Epsilon || m_letters[c] < RowSize());
		}
		memcpy(m_final, s.m_final, m.finalTableSize * sizeof(*m_final));
		memcpy(m_finalIndex, s.m_finalIndex, m.statesCount * sizeof(*m_finalIndex));

		m.initial = IndexToState(s.StateIndex(s.m.initial));

		for (size_t st = 0; st != m.statesCount; ++st) {
			size_t oldstate = s.IndexToState(st);
			size_t newstate = IndexToState(st);
			Header(newstate) = s.Header(oldstate);
			const typename Scanner<AnotherRelocation, Shortcutting>::Transition* os
				= reinterpret_cast<const typename Scanner<AnotherRelocation, Shortcutting>::Transition*>(oldstate);
			Transition* ns = reinterpret_cast<Transition*>(newstate);

			for (size_t let = 0; let != LettersCount(); ++let) {
				size_t destIndex = s.StateIndex(AnotherRelocation::Go(oldstate, os[let + s.HEADER_SIZE]));
				Transition tr = Relocation::Diff(newstate, IndexToState(destIndex));
				ns[let + HEADER_SIZE] = tr;
				Y_ASSERT(Relocation::Go(newstate, tr) >= (size_t)m_transitions);
				Y_ASSERT(Relocation::Go(newstate, tr) < (size_t)(m_transitions + RowSize()*Size()));
			}
		}
	}


	size_t IndexToState(size_t stateIndex) const
	{
		return reinterpret_cast<size_t>(m_transitions + stateIndex * RowSize());
	}

	void SetJump(size_t oldState, Char c, size_t newState, unsigned long /*payload*/ = 0)
	{
		Y_ASSERT(m_buffer);
		Y_ASSERT(oldState < m.statesCount);
		Y_ASSERT(newState < m.statesCount);

		m_transitions[oldState * RowSize() + m_letters[c]]
			= Relocation::Diff(IndexToState(oldState), IndexToState(newState));
	}

	unsigned long RemapAction(unsigned long action) { return action; }

	void SetInitial(size_t state)
	{
		Y_ASSERT(m_buffer);
		m.initial = IndexToState(state);
	}

	void SetTag(size_t state, size_t value)
	{
		Y_ASSERT(m_buffer);
		Header(IndexToState(state)).Common.Flags = value;
	}

	// Fill shortcut masks for all the states
	void BuildShortcuts()
	{
		Y_ASSERT(m_buffer);

		// Build the mapping from letter classes to characters
		TVector< TVector<char> > letters(RowSize());
		for (unsigned ch = 0; ch != 1 << (sizeof(char)*8); ++ch)
			letters[m_letters[ch]].push_back(ch);

		// Loop through all states in the transition table and
		// check if it is possible to setup shortcuts
		for (size_t i = 0; i != Size(); ++i) {
			State st = IndexToState(i);
			ScannerRowHeader& header = Header(st);
			Shortcutting::SetNoExit(header);
			size_t ind = 0;
			size_t let = HEADER_SIZE;
			for (; let != LettersCount() + HEADER_SIZE; ++let) {
				// Check if the transition is not the same state
				if (Relocation::Go(st, reinterpret_cast<const Transition*>(st)[let]) != st) {
					if (ind + letters[let].size() > Shortcutting::ExitMaskCount)
						break;
					// For each character setup a mask
					for (auto&& character : letters[let]) {
						Shortcutting::SetMask(header, ind, character);
						++ind;
					}
				}
			}

			if (let != LettersCount() + HEADER_SIZE) {
				// Not enough space in ExitMasks, so reset all masks (which leads to bypassing the optimization)
				Shortcutting::SetNoShortcut(header);
			}
			// Fill the rest of the shortcut masks with the last used mask
			Shortcutting::FinishMasks(header, ind);
		}
	}

	// Fills final states table and builds shortcuts if possible
	void FinishBuild()
	{
		Y_ASSERT(m_buffer);
		auto finalWriter = m_final;
		for (size_t state = 0; state != Size(); ++state) {
			m_finalIndex[state] = finalWriter - m_final;
			if (Header(IndexToState(state)).Common.Flags & FinalFlag)
				*finalWriter++ = 0;
			*finalWriter++ = static_cast<size_t>(-1);
		}
		BuildShortcuts();
	}

	size_t AcceptedRegexpsCount(size_t idx) const
	{
		const size_t* b = m_final + m_finalIndex[idx];
		const size_t* e = b;
		while (*e != End)
			++e;
		return e - b;
	}

	template <class Scanner>
	friend void Pire::BuildScanner(const Fsm&, Scanner&);

	typedef State InternalState; // Needed for agglutination
	friend class ScannerGlueCommon<Scanner>;
	friend class ScannerGlueTask<Scanner>;

	template<class AnotherRelocation, class AnotherShortcutting>
	friend class Scanner;

    friend struct ScannerSaver;

#ifndef PIRE_DEBUG
	friend struct AlignedRunner< Scanner<Relocation, Shortcutting> >;
#endif
};

// Helper class for Save/Load partial specialization
struct ScannerSaver {
	template<class Shortcutting>
	static void SaveScanner(const Scanner<Relocatable, Shortcutting>& scanner, yostream* s)
	{
		typedef Scanner<Relocatable, Shortcutting> ScannerType;

		typename ScannerType::Locals mc = scanner.m;
		mc.initial -= reinterpret_cast<size_t>(scanner.m_transitions);
		SavePodType(s, Pire::Header(ScannerIOTypes::Scanner, sizeof(mc)));
		Impl::AlignSave(s, sizeof(Pire::Header));
		SavePodType(s, mc);
		Impl::AlignSave(s, sizeof(mc));
		SavePodType(s, scanner.Empty());
		Impl::AlignSave(s, sizeof(scanner.Empty()));
		if (!scanner.Empty())
			Impl::AlignedSaveArray(s, scanner.m_buffer.Get(), scanner.BufSize());
	}

	template<class Shortcutting>
	static void LoadScanner(Scanner<Relocatable, Shortcutting>& scanner, yistream* s)
	{
		typedef Scanner<Relocatable, Shortcutting> ScannerType;

		Scanner<Relocatable, Shortcutting> sc;
		Impl::ValidateHeader(s, ScannerIOTypes::Scanner, sizeof(sc.m));
		LoadPodType(s, sc.m);
		Impl::AlignLoad(s, sizeof(sc.m));
		if (Shortcutting::Signature != sc.m.shortcuttingSignature)
			throw Error("This scanner has different shortcutting type");
		bool empty;
		LoadPodType(s, empty);
		Impl::AlignLoad(s, sizeof(empty));

		if (empty) {
			sc.Alias(ScannerType::Null());
		} else {
			sc.m_buffer = TArrayHolder<char>(new char[sc.BufSize()]);
			Impl::AlignedLoadArray(s, sc.m_buffer.Get(), sc.BufSize());
			sc.Markup(sc.m_buffer.Get());
			sc.m.initial += reinterpret_cast<size_t>(sc.m_transitions);
		}
		scanner.Swap(sc);
	}

	// TODO: implement more effective serialization
	// of nonrelocatable scanner if necessary

	template<class Shortcutting>
	static void SaveScanner(const Scanner<Nonrelocatable, Shortcutting>& scanner, yostream* s)
	{
		Scanner<Relocatable, Shortcutting>(scanner).Save(s);
	}

	template<class Shortcutting>
	static void LoadScanner(Scanner<Nonrelocatable, Shortcutting>& scanner, yistream* s)
	{
		Scanner<Relocatable, Shortcutting> rs;
		rs.Load(s);
		Scanner<Nonrelocatable, Shortcutting>(rs).Swap(scanner);
	}
};


template<class Relocation, class Shortcutting>
void Scanner<Relocation, Shortcutting>::Save(yostream* s) const
{
	ScannerSaver::SaveScanner(*this, s);
}

template<class Relocation, class Shortcutting>
void Scanner<Relocation, Shortcutting>::Load(yistream* s)
{
	ScannerSaver::LoadScanner(*this, s);
}

// Shortcutting policy that checks state exit masks
template <size_t MaskCount>
class ExitMasks {
private:
	enum {
		NO_SHORTCUT_MASK = 1, // the state doesn't have shortcuts
		NO_EXIT_MASK  =    2  // the state has only transtions to itself (we can stop the scan)
	};

	template<class ScannerRowHeader, unsigned N>
	struct MaskCheckerBase {
		static PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
		bool Check(const ScannerRowHeader& hdr, size_t alignOffset, Word chunk)
		{
			Word mask = CheckBytes(hdr.Mask(N, alignOffset), chunk);
			for (int i = N-1; i >= 0; --i) {
				mask = Or(mask, CheckBytes(hdr.Mask(i, alignOffset), chunk));
			}
			return !IsAnySet(mask);
		}

		static PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
		const Word* DoRun(const ScannerRowHeader& hdr, size_t alignOffset, const Word* begin, const Word* end)
		{
			for (; begin != end && Check(hdr, alignOffset, ToLittleEndian(*begin)); ++begin) {}
			return begin;
		}
	};

	template<class ScannerRowHeader, unsigned N, unsigned Nmax>
	struct MaskChecker : MaskCheckerBase<ScannerRowHeader, N>  {
		typedef MaskCheckerBase<ScannerRowHeader, N> Base;
		typedef MaskChecker<ScannerRowHeader, N+1, Nmax> Next;

		static PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
		const Word* Run(const ScannerRowHeader& hdr, size_t alignOffset, const Word* begin, const Word* end)
		{
			if (hdr.Mask(N) == hdr.Mask(N + 1))
				return Base::DoRun(hdr, alignOffset, begin, end);
			else
				return Next::Run(hdr, alignOffset, begin, end);
		}
	};

	template<class ScannerRowHeader, unsigned N>
	struct MaskChecker<ScannerRowHeader, N, N> : MaskCheckerBase<ScannerRowHeader, N>  {
		typedef MaskCheckerBase<ScannerRowHeader, N> Base;

		static PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
		const Word* Run(const ScannerRowHeader& hdr, size_t alignOffset, const Word* begin, const Word* end)
		{
			return Base::DoRun(hdr, alignOffset, begin, end);
		}
	};

	// Compares the ExitMask[0] value without SSE reads which seems to be more optimal
	template <class Relocation>
	static PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	bool CheckFirstMask(const Scanner<Relocation, ExitMasks<MaskCount> >& scanner, typename Scanner<Relocation, ExitMasks<MaskCount> >::State state, size_t val)
	{
		return (scanner.Header(state).Mask(0) == val);
	}

public:

	static const size_t ExitMaskCount = MaskCount;
	static const size_t Signature = 0x2000 + MaskCount;

	template <class Scanner>
	struct ExtendedRowHeader {
	private:
		/// In order to allow transition table to be aligned at sizeof(size_t) instead of
		/// sizeof(Word) and still be able to read Masks at Word-aligned addresses each mask
		/// occupies 2x space and only properly aligned part of it is read
		enum {
			SizeTInMaxSizeWord = sizeof(MaxSizeWord) / sizeof(size_t),
			MaskSizeInSizeT = 2 * SizeTInMaxSizeWord,
		};

	public:
		static const size_t ExitMaskCount = MaskCount;

		inline
		const Word& Mask(size_t i, size_t alignOffset) const
		{
			Y_ASSERT(i < ExitMaskCount);
			Y_ASSERT(alignOffset < SizeTInMaxSizeWord);
			const Word* p = (const Word*)(ExitMasksArray + alignOffset + MaskSizeInSizeT * i);
			Y_ASSERT(IsAligned(p, sizeof(Word)));
			return *p;
		}

		PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
		size_t Mask(size_t i) const
		{
			Y_ASSERT(i < ExitMaskCount);
			return ExitMasksArray[MaskSizeInSizeT*i];
		}

		void SetMask(size_t i, size_t val)
		{
			for (size_t j = 0; j < MaskSizeInSizeT; ++j)
				ExitMasksArray[MaskSizeInSizeT*i + j] = val;
		}

		ExtendedRowHeader()
		{
			for (size_t i = 0; i < ExitMaskCount; ++i)
				SetMask(i, NO_SHORTCUT_MASK);
		}

		template <class OtherScanner>
		ExtendedRowHeader& operator =(const ExtendedRowHeader<OtherScanner>& other)
		{
			PIRE_STATIC_ASSERT(ExitMaskCount == ExtendedRowHeader<OtherScanner>::ExitMaskCount);
			Common = other.Common;
			for (size_t i = 0; i < ExitMaskCount; ++i)
				SetMask(i, other.Mask(i));
			return *this;
		}

	private:
		/// If this state loops for all letters except particular set
		/// (common thing when matching something like /.*[Aa]/),
		/// each ExitMask contains that letter in each byte of size_t.
		///
		/// These masks are most commonly used for fast forwarding through parts
		/// of the string matching /.*/ somewhere in the middle regexp.
		size_t ExitMasksArray[ExitMaskCount * MaskSizeInSizeT];

	public:
		typename Scanner::CommonRowHeader Common;
	};

	template <class Header>
	static void SetNoExit(Header& header)
	{
		header.SetMask(0, NO_EXIT_MASK);
	}

	template <class Header>
	static void SetNoShortcut(Header& header)
	{
		header.SetMask(0, NO_SHORTCUT_MASK);
	}

	template <class Header>
	static void SetMask(Header& header, size_t ind, char c)
	{
		header.SetMask(ind, FillSizeT(c));
	}

	template <class Header>
	static void FinishMasks(Header& header, size_t ind)
	{
		if (ind == 0)
			ind = 1;
		// Fill the rest of the shortcut masks with the last used mask
		size_t lastMask = header.Mask(ind - 1);
		while (ind != ExitMaskCount) {
			header.SetMask(ind, lastMask);
			++ind;
		}
	}

	template <class Relocation>
	static PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	bool NoExit(const Scanner<Relocation, ExitMasks<MaskCount> >& scanner, typename Scanner<Relocation, ExitMasks<MaskCount> >::State state)
	{
		return CheckFirstMask(scanner, state, NO_EXIT_MASK);
	}

	template <class Relocation>
	static PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	bool NoShortcut(const Scanner<Relocation, ExitMasks<MaskCount> >& scanner, typename Scanner<Relocation, ExitMasks<MaskCount> >::State state)
	{
		return CheckFirstMask(scanner, state, NO_SHORTCUT_MASK);
	}

	template <class Relocation>
	static PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	const Word* Run(const Scanner<Relocation, ExitMasks<MaskCount> >& scanner, typename Scanner<Relocation, ExitMasks<MaskCount> >::State state, size_t alignOffset, const Word* begin, const Word* end)
	{
		return MaskChecker<typename Scanner<Relocation, ExitMasks<MaskCount> >::ScannerRowHeader, 0, MaskCount - 1>::Run(scanner.Header(state), alignOffset, begin, end);
	}

};


// Shortcutting policy that doesn't do shortcuts
struct NoShortcuts {

	static const size_t ExitMaskCount = 0;
	static const size_t Signature = 0x1000;

	template <class Scanner>
	struct ExtendedRowHeader {
		typename Scanner::CommonRowHeader Common;

		template <class OtherScanner>
		ExtendedRowHeader& operator =(const ExtendedRowHeader<OtherScanner>& other)
		{
			PIRE_STATIC_ASSERT(sizeof(ExtendedRowHeader) == sizeof(ExtendedRowHeader<OtherScanner>));
			Common = other.Common;
			return *this;
		}
	};

	template <class Header>
	static void SetNoExit(Header&) {}

	template <class Header>
	static void SetNoShortcut(Header&) {}

	template <class Header>
	static void SetMask(Header&, size_t, char) {}

	template <class Header>
	static void FinishMasks(Header&, size_t) {}

	template <class Relocation>
	static PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	bool NoExit(const Scanner<Relocation, NoShortcuts>&, typename Scanner<Relocation, NoShortcuts>::State)
	{
		// Cannot exit prematurely
		return false;
	}

	template <class Relocation>
	static PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	bool NoShortcut(const Scanner<Relocation, NoShortcuts>&, typename Scanner<Relocation, NoShortcuts>::State)
	{
		// There's no shortcut regardless of the state
		return true;
	}

	template <class Relocation>
	static PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	const Word* Run(const Scanner<Relocation, NoShortcuts>&, typename Scanner<Relocation, NoShortcuts>::State, size_t, const Word* begin, const Word*)
	{
		// Stop shortcutting right at the beginning
		return begin;
	}
};

#ifndef PIRE_DEBUG

// The purpose of this template is to produce a number of ProcessChunk() calls
// instead of writing for(...){ProcessChunk()} loop that GCC refuses to unroll.
// Manually unrolled code proves to be faster
template <class Scanner, unsigned Count>
struct MultiChunk {
	// Process Word-sized chunk which consist of >=1 size_t-sized chunks
	template<class Pred>
	static PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	Action Process(const Scanner& scanner, typename Scanner::State& state, const size_t* p, Pred pred)
	{
		if (RunChunk(scanner, state, p, 0, sizeof(void*), pred) == Continue)
			return MultiChunk<Scanner, Count-1>::Process(scanner, state, ++p, pred);
		else
			return Stop;
	}
};

template <class Scanner>
struct MultiChunk<Scanner, 0> {
	// Process Word-sized chunk which consist of >=1 size_t-sized chunks
	template<class Pred>
	static PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	Action Process(const Scanner&, typename Scanner::State, const size_t*, Pred)
	{
		return Continue;
	}
};

// Efficiently runs a scanner through size_t-aligned memory range
template<class Relocation, class Shortcutting>
struct AlignedRunner< Scanner<Relocation, Shortcutting> > {
private:
	typedef Scanner<Relocation, Shortcutting> ScannerType;

	// Processes Word-sized chuck of memory (depending on the platform a Word might
	// consist of multiple size_t chuncks)
	template <class Pred>
	static PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	Action RunMultiChunk(const ScannerType& scanner, typename ScannerType::State& st, const size_t* begin, Pred pred)
	{
		return MultiChunk<ScannerType, sizeof(Word)/sizeof(size_t)>::Process(scanner, st, begin, pred);
	}

	// Asserts if the scanner changes state while processing the byte range that is
	// supposed to be skipped by a shortcut
	static void ValidateSkip(const ScannerType& scanner, typename ScannerType::State st, const char* begin, const char* end)
	{
		typename ScannerType::State stateBefore = st;
		for (const char* pos = begin; pos != end; ++pos) {
			Step(scanner, st, (unsigned char)*pos);
			Y_ASSERT(st == stateBefore);
		}
	}

public:

	template<class Pred>
	static inline PIRE_HOT_FUNCTION
	Action RunAligned(const ScannerType& scanner, typename ScannerType::State& st, const size_t* begin, const size_t* end , Pred pred)
	{
		typename ScannerType::State state = st;
		const Word* head = AlignUp((const Word*) begin, sizeof(Word));
		const Word* tail = AlignDown((const Word*) end, sizeof(Word));
		for (; begin != (const size_t*) head && begin != end; ++begin)
			if (RunChunk(scanner, state, begin, 0, sizeof(void*), pred) == Stop) {
				st = state;
				return Stop;
			}

		if (begin == end) {
			st = state;
			return Continue;
		}
		if (Shortcutting::NoExit(scanner, state)) {
			st = state;
			return pred(scanner, state, ((const char*) end));
		}

		// Row size should be a multiple of MaxSizeWord size. Then alignOffset is the same for any state
		Y_ASSERT((scanner.RowSize()*sizeof(typename ScannerType::Transition)) % sizeof(MaxSizeWord) == 0);
		size_t alignOffset = (AlignUp((size_t)scanner.m_transitions, sizeof(Word)) - (size_t)scanner.m_transitions) / sizeof(size_t);

		bool noShortcut = Shortcutting::NoShortcut(scanner, state);

		while (true) {
			// Do normal processing until a shortcut is possible
			while (noShortcut && head != tail) {
				if (RunMultiChunk(scanner, state, (const size_t*)head, pred) == Stop) {
					st = state;
					return Stop;
				}
				++head;
				noShortcut = Shortcutting::NoShortcut(scanner, state);
			}
			if (head == tail)
				break;

			if (Shortcutting::NoExit(scanner, state)) {
				st = state;
				return pred(scanner, state, ((const char*) end));
			}

			// Do fast forwarding while it is possible
			const Word* skipEnd = Shortcutting::Run(scanner, state, alignOffset, head, tail);
			PIRE_IF_CHECKED(ValidateSkip(scanner, state, (const char*)head, (const char*)skipEnd));
			head = skipEnd;
			noShortcut = true;
		}

		for (size_t* p = (size_t*) tail; p != end; ++p) {
			if (RunChunk(scanner, state, p, 0, sizeof(void*), pred) == Stop) {
				st = state;
				return Stop;
			}
		}

		st = state;
		return Continue;
	}
};

#endif

template<class Scanner>
class ScannerGlueTask: public ScannerGlueCommon<Scanner> {
public:
	typedef ScannerGlueCommon<Scanner> Base;
	typedef typename Base::State State;
	using Base::Lhs;
	using Base::Rhs;
	using Base::Sc;
	using Base::Letters;

	typedef GluedStateLookupTable<256*1024, typename Scanner::State> InvStates;

	ScannerGlueTask(const Scanner& lhs, const Scanner& rhs)
		: ScannerGlueCommon<Scanner>(lhs, rhs, LettersEquality<Scanner>(lhs.m_letters, rhs.m_letters))
	{
	}

	void AcceptStates(const TVector<State>& states)
	{
		// Make up a new scanner and fill in the final table

		size_t finalTableSize = 0;
		for (auto&& i : states)
			finalTableSize += RangeLen(Lhs().AcceptedRegexps(i.first)) + RangeLen(Rhs().AcceptedRegexps(i.second));
		this->SetSc(THolder<Scanner>(new Scanner));
		Sc().Init(states.size(), Letters(), finalTableSize, size_t(0), Lhs().RegexpsCount() + Rhs().RegexpsCount());

		auto finalWriter = Sc().m_final;
		for (size_t state = 0; state != states.size(); ++state) {
			Sc().m_finalIndex[state] = finalWriter - Sc().m_final;
			finalWriter = Shift(Lhs().AcceptedRegexps(states[state].first), 0, finalWriter);
			finalWriter = Shift(Rhs().AcceptedRegexps(states[state].second), Lhs().RegexpsCount(), finalWriter);
			*finalWriter++ = static_cast<size_t>(-1);

			Sc().SetTag(state, ((Lhs().Final(states[state].first) || Rhs().Final(states[state].second)) ? Scanner::FinalFlag : 0)
				| ((Lhs().Dead(states[state].first) && Rhs().Dead(states[state].second)) ? Scanner::DeadFlag : 0));
		}
	}

	void Connect(size_t from, size_t to, Char letter) { Sc().SetJump(from, letter, to); }

	const Scanner& Success()
	{
		Sc().BuildShortcuts();
		return Sc();
	}

private:
	template<class Iter>
	size_t RangeLen(ypair<Iter, Iter> range) const
	{
		return std::distance(range.first, range.second);
	}

	template<class Iter, class OutIter>
	OutIter Shift(ypair<Iter, Iter> range, size_t shift, OutIter out) const
	{
		for (; range.first != range.second; ++range.first, ++out)
			*out = *range.first + shift;
		return out;
	}
};

}


template<class Relocation, class Shortcutting>
struct StDumper< Impl::Scanner<Relocation, Shortcutting> > {

	typedef Impl::Scanner<Relocation, Shortcutting> ScannerType;

	StDumper(const ScannerType& sc, typename ScannerType::State st): m_sc(&sc), m_st(st) {}

	void Dump(yostream& stream) const
	{
		stream << m_sc->StateIndex(m_st);
		if (m_sc->Final(m_st))
			stream << " [final]";
		if (m_sc->Dead(m_st))
			stream << " [dead]";
	}
private:
	const ScannerType* m_sc;
	typename ScannerType::State m_st;
};


template<class Relocation, class Shortcutting>
Impl::Scanner<Relocation, Shortcutting> Impl::Scanner<Relocation, Shortcutting>::Glue(const Impl::Scanner<Relocation, Shortcutting>& lhs, const Impl::Scanner<Relocation, Shortcutting>& rhs, size_t maxSize /* = 0 */)
{
	if (lhs.Empty())
		return rhs;
	if (rhs.Empty())
		return lhs;

	static const size_t DefMaxSize = 80000;
	Impl::ScannerGlueTask< Impl::Scanner<Relocation, Shortcutting> > task(lhs, rhs);
	return Impl::Determine(task, maxSize ? maxSize : DefMaxSize);
}


/**
 * A compiled multiregexp.
 * Can only find out whether a string matches the regexps or not,
 * but takes O( str.length() ) time.
 *
 * In addition, multiple scanners can be agglutinated together,
 * producting a scanner which can be used for checking
 * strings against several regexps in a single pass.
 */
typedef Impl::Scanner<Impl::Relocatable, Impl::ExitMasks<2> > Scanner;
typedef Impl::Scanner<Impl::Relocatable, Impl::NoShortcuts> ScannerNoMask;

/**
 * Same as above, but does not allow relocation or mmap()-ing.
 * On the other hand, runs almost twice as fast as the Scanner.
 */
typedef Impl::Scanner<Impl::Nonrelocatable, Impl::ExitMasks<2> > NonrelocScanner;
typedef Impl::Scanner<Impl::Nonrelocatable, Impl::NoShortcuts> NonrelocScannerNoMask;

}

namespace std {
	inline void swap(Pire::Scanner& a, Pire::Scanner& b) {
		a.Swap(b);
	}

	inline void swap(Pire::NonrelocScanner& a, Pire::NonrelocScanner& b) {
		a.Swap(b);
	}
}


#endif
