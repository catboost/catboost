/*
 * run.h -- routines for running scanners on strings.
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


#ifndef PIRE_RE_SCANNER_H
#define PIRE_RE_SCANNER_H

#include <contrib/libs/pire/pire/stub/stl.h>
#include <contrib/libs/pire/pire/stub/memstreams.h>
#include <contrib/libs/pire/pire/scanners/pair.h>

#include "platform.h"
#include "defs.h"

#include <string>

namespace Pire {

	template<class Scanner>
	struct StDumper {
		StDumper(const Scanner& sc, typename Scanner::State st): m_sc(&sc), m_st(st) {}
		void Dump(yostream& stream) const { stream << m_sc->StateIndex(m_st) << (m_sc->Final(m_st) ? " [final]" : ""); }
	private:
		const Scanner* m_sc;
		typename Scanner::State m_st;
	};

	template<class Scanner> StDumper<Scanner> StDump(const Scanner& sc, typename Scanner::State st) { return StDumper<Scanner>(sc, st); }
	template<class Scanner> yostream& operator << (yostream& stream, const StDumper<Scanner>& stdump) { stdump.Dump(stream); return stream; }
}

namespace Pire {

template<class Scanner>
PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
void Step(const Scanner& scanner, typename Scanner::State& state, Char ch)
{
	Y_ASSERT(ch < MaxCharUnaligned);
	typename Scanner::Action a = scanner.Next(state, ch);
	scanner.TakeAction(state, a);
}

namespace Impl {

	enum Action { Continue, Stop };

	template<class Scanner>
	struct RunPred {
		PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
		Action operator()(const Scanner&, const typename Scanner::State&, const char*) const { return Continue; }
	};
	
	template<class Scanner>
	struct ShortestPrefixPred {
		explicit ShortestPrefixPred(const char*& pos): m_pos(&pos) {}

		PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
		Action operator()(const Scanner& sc, const typename Scanner::State& st, const char* pos) const
		{
			if (sc.Final(st)) {
				*m_pos = pos;
				return Stop;
			} else {
				return (sc.Dead(st) ? Stop : Continue);
			}
		}
	private:
		const char** m_pos;
	};
	
	template<class Scanner>
	struct LongestPrefixPred {
		explicit LongestPrefixPred(const char*& pos): m_pos(&pos) {}
		
		PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
		Action operator()(const Scanner& sc, const typename Scanner::State& st, const char* pos) const
		{
			if (sc.Final(st))
				*m_pos = pos;
			return (sc.Dead(st) ? Stop : Continue);
		}
	private:
		const char** m_pos;
	};

}

#ifndef PIRE_DEBUG

namespace Impl {

	template<class Scanner, class Pred>
	PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	Action SafeRunChunk(const Scanner& scanner, typename Scanner::State& state, const size_t* p, size_t pos, size_t size, Pred pred)
	{
		Y_ASSERT(pos <= sizeof(size_t));
		Y_ASSERT(size <= sizeof(size_t));
		Y_ASSERT(pos + size <= sizeof(size_t));

		if (PIRE_UNLIKELY(size == 0))
			return Continue;

		const char* ptr = (const char*) p + pos;
		for (; size--; ++ptr) {
			Step(scanner, state, (unsigned char) *ptr);
			if (pred(scanner, state, ptr + 1) == Stop)
				return Stop;
		}
		return Continue;
	}

	/// Effectively runs a scanner on a short data chunk, fit completely into one machine word.
	template<class Scanner, class Pred>
	PIRE_FORCED_INLINE PIRE_HOT_FUNCTION
	Action RunChunk(const Scanner& scanner, typename Scanner::State& state, const size_t* p, size_t pos, size_t size, Pred pred)
	{
		Y_ASSERT(pos <= sizeof(size_t));
		Y_ASSERT(size <= sizeof(size_t));
		Y_ASSERT(pos + size <= sizeof(size_t));

		if (PIRE_UNLIKELY(size == 0))
			return Continue;

		size_t chunk = Impl::ToLittleEndian(*p) >> 8*pos;
		const char* ptr = (const char*) p + pos + size + 1;

		for (size_t i = size; i != 0; --i) {
			Step(scanner, state, chunk & 0xFF);
			if (pred(scanner, state, ptr - i) == Stop)
				return Stop;
			chunk >>= 8;
		}
		return Continue;
	}

	template<class Scanner>
	struct AlignedRunner {

		// Generic version for LongestPrefix()/ShortestPrefix() impelementations
		template<class Pred>
		static inline PIRE_HOT_FUNCTION
		Action RunAligned(const Scanner& scanner, typename Scanner::State& state, const size_t* begin, const size_t* end, Pred stop)
		{
			typename Scanner::State st = state;
			Action ret = Continue;
			for (; begin != end && (ret = RunChunk(scanner, st, begin, 0, sizeof(void*), stop)) == Continue; ++begin)
				;
			state = st;
			return ret;
		}

		// A special version for Run() impelementation that skips predicate checks
		static inline PIRE_HOT_FUNCTION
		Action RunAligned(const Scanner& scanner, typename Scanner::State& state, const size_t* begin, const size_t* end, RunPred<Scanner>)
		{
			typename Scanner::State st = state;
			for (; begin != end; ++begin) {
				size_t chunk = *begin;
				for (size_t i = sizeof(chunk); i != 0; --i) {
					Step(scanner, st, chunk & 0xFF);
					chunk >>= 8;
				}
			}
			state = st;
			return Continue;
		}
	};

	/// The main function: runs a scanner through given memory range.
	template<class Scanner, class Pred>
	inline void DoRun(const Scanner& scanner, typename Scanner::State& st, TStringBuf str, Pred pred)
	{

		const size_t* head = reinterpret_cast<const size_t*>((reinterpret_cast<uintptr_t>(str.begin())) & ~(sizeof(size_t)-1));
		const size_t* tail = reinterpret_cast<const size_t*>((reinterpret_cast<uintptr_t>(str.end())) & ~(sizeof(size_t)-1));

		size_t headSize = (sizeof(size_t) - (str.begin() - (const char*)head)); // The distance from @p begin to the end of the word containing @p begin
		size_t tailSize = str.end() - (const char*) tail; // The distance from the beginning of the word containing @p end to the @p end

		Y_ASSERT(headSize >= 1 && headSize <= sizeof(size_t));
		Y_ASSERT(tailSize < sizeof(size_t));

		if (head == tail) {
			Impl::SafeRunChunk(scanner, st, head, sizeof(size_t) - headSize, str.end() - str.begin(), pred);
			return;
		}

		// st is passed by reference to this function. If we use it directly on each step the compiler will have to
		// update it in memory because of pointer aliasing assumptions. Copying it into a local var allows the
		// compiler to store it in a register. This saves some instructions and cycles
		typename Scanner::State state = st;

		if (str.begin() != (const char*) head) {
			if (Impl::RunChunk(scanner, state, head, sizeof(size_t) - headSize, headSize, pred) == Stop) {
				st = state;
				return;
			}
			++head;
		}

		if (Impl::AlignedRunner<Scanner>::RunAligned(scanner, state, head, tail, pred) == Stop) {
			st = state;
			return;
		}

		if (tailSize)
			Impl::SafeRunChunk(scanner, state, tail, 0, tailSize, pred);

		st = state;
	}

}

/// Runs two scanners through given memory range simultaneously.
/// This is several percent faster than running them independently.
template<class Scanner1, class Scanner2>
inline void Run(const Scanner1& scanner1, const Scanner2& scanner2, typename Scanner1::State& state1, typename Scanner2::State& state2, TStringBuf str)
{
	typedef ScannerPair<Scanner1, Scanner2> Scanners;
	Scanners pair(scanner1, scanner2);
	typename Scanners::State states(state1, state2);
	Run(pair, states, str);
	state1 = states.first;
	state2 = states.second;
}

#else

namespace Impl {
	/// A debug version of all Run() methods.
	template<class Scanner, class Pred>
	inline void DoRun(const Scanner& scanner, typename Scanner::State& state, const char* begin, const char* end, Pred pred)
	{
		Cdbg << "Running regexp on string " << ystring(begin, ymin(end - begin, static_cast<ptrdiff_t>(100u))) << Endl;
		Cdbg << "Initial state " << StDump(scanner, state) << Endl;

		if (pred(scanner, state, begin) == Stop) {
			Cdbg << " exiting" << Endl;
			return;
		}

		for (; begin != end; ++begin) {
			Step(scanner, state, (unsigned char)*begin);
			Cdbg << *begin << " => state " << StDump(scanner, state) << Endl;
			if (pred(scanner, state, begin + 1) == Stop) {
				Cdbg << " exiting" << Endl;
				return;
			}
		}
	}
}

#endif
	
template<class Scanner>
void Run(const Scanner& sc, typename Scanner::State& st, TStringBuf str)
{
	Impl::DoRun(sc, st, str, Impl::RunPred<Scanner>());
}

template<class Scanner>
void Run(const Scanner& sc, typename Scanner::State& st, const char* begin, const char* end)
{
	Run(sc, st, TStringBuf(begin, end));
}

/// Returns default constructed string_view{} if there is no matching prefix
/// Returns str.substr(0, 0) if matching prefix is empty
template<class Scanner>
std::string_view LongestPrefix(const Scanner& sc, std::string_view str, bool throughBeginMark = false, bool throughEndMark = false)
{
	typename Scanner::State st;
	sc.Initialize(st);
	if (throughBeginMark)
		Pire::Step(sc, st, BeginMark);
	const char* pos = (sc.Final(st) ? str.data() : nullptr);
	Impl::DoRun(sc, st, str, Impl::LongestPrefixPred<Scanner>(pos));
	if (throughEndMark) {
		Pire::Step(sc, st, EndMark);
		if (sc.Final(st))
			pos = str.data() + str.size();
	}
	return pos ? str.substr(0, pos - str.data()) : std::string_view{};
}

template<class Scanner>
const char* LongestPrefix(const Scanner& sc, const char* begin, const char* end, bool throughBeginMark = false, bool throughEndMark = false)
{
	auto prefix = LongestPrefix(sc, std::string_view(begin, end - begin), throughBeginMark, throughEndMark);
	return prefix.data() + prefix.size();
}

/// Returns default constructed string_view{} if there is no matching prefix
/// Returns str.substr(0, 0) if matching prefix is empty
template<class Scanner>
std::string_view ShortestPrefix(const Scanner& sc, std::string_view str, bool throughBeginMark = false, bool throughEndMark = false)
{
	typename Scanner::State st;
	sc.Initialize(st);
	if (throughBeginMark)
		Pire::Step(sc, st, BeginMark);
	if (sc.Final(st))
		return str.substr(0, 0);
	const char* pos = nullptr;
	Impl::DoRun(sc, st, str, Impl::ShortestPrefixPred<Scanner>(pos));
	if (throughEndMark) {
		Pire::Step(sc, st, EndMark);
		if (sc.Final(st) && !pos)
			pos = str.data() + str.size();
	}
	return pos ? str.substr(0, pos - str.data()) : std::string_view{};
}

template<class Scanner>
const char* ShortestPrefix(const Scanner& sc, const char* begin, const char* end, bool throughBeginMark = false, bool throughEndMark = false)
{
    auto prefix = ShortestPrefix(sc, std::string_view(begin, end - begin), throughBeginMark, throughEndMark);
    return prefix.data() + prefix.size();
}

	
/// The same as above, but scans string in reverse direction
/// (consider using Fsm::Reverse() for using in this function).
/// Returns default constructed string_view{} if there is no matching suffix
/// Returns str.substr(str.size(), 0) if matching suffix is empty
template<class Scanner>
inline std::string_view LongestSuffix(const Scanner& scanner, std::string_view str, bool throughEndMark = false, bool throughBeginMark = false)
{
	typename Scanner::State state;
	scanner.Initialize(state);
	if (throughEndMark)
		Step(scanner, state, EndMark);
	PIRE_IFDEBUG(Cdbg << "Running LongestSuffix on string " << ystring(str) << Endl);
	PIRE_IFDEBUG(Cdbg << "Initial state " << StDump(scanner, state) << Endl);

	std::string_view suffix{};
	auto begin = str.data() + str.size();
	while (begin != str.data() && !scanner.Dead(state)) {
		if (scanner.Final(state))
			suffix = str.substr(begin - str.data());
		--begin;
		Step(scanner, state, (unsigned char)*begin);
		PIRE_IFDEBUG(Cdbg << *begin << " => state " << StDump(scanner, state) << Endl);
	}
	if (scanner.Final(state))
		suffix = str.substr(begin - str.data());
	if (throughBeginMark) {
		Step(scanner, state, BeginMark);
		if (scanner.Final(state))
			suffix = str.substr(begin - str.data());
	}
	return suffix;
}

template<class Scanner>
inline const char* LongestSuffix(const Scanner& scanner, const char* rbegin, const char* rend, bool throughEndMark = false, bool throughBeginMark = false) {
	auto suffix = LongestSuffix(scanner, std::string_view(rend + 1, rbegin - rend), throughEndMark, throughBeginMark);
    return suffix.data() ? suffix.data() - 1 : nullptr;
}

/// The same as above, but scans string in reverse direction
/// Returns default constructed string_view{} if there is no matching suffix
/// Returns str.substr(str.size(), 0) if matching suffix is empty
template<class Scanner>
inline std::string_view ShortestSuffix(const Scanner& scanner, std::string_view str, bool throughEndMark = false, bool throughBeginMark = false)
{
	auto begin = str.data() + str.size();
	typename Scanner::State state;
	scanner.Initialize(state);
	if (throughEndMark)
		Step(scanner, state, EndMark);
	PIRE_IFDEBUG(Cdbg << "Running ShortestSuffix on string " << ystring(str) << Endl);
	PIRE_IFDEBUG(Cdbg << "Initial state " << StDump(scanner, state) << Endl);

	while (begin != str.data() && !scanner.Final(state) && !scanner.Dead(state)) {
		--begin;
		scanner.Next(state, (unsigned char)*begin);
		PIRE_IFDEBUG(Cdbg << *rbegin << " => state " << StDump(scanner, state) << Endl);
	}
	if (throughBeginMark)
		Step(scanner, state, BeginMark);
	return scanner.Final(state) ? str.substr(begin - str.data()) : std::string_view{};
}

template<class Scanner>
inline const char* ShortestSuffix(const Scanner& scanner, const char* rbegin, const char* rend, bool throughEndMark = false, bool throughBeginMark = false) {
	auto suffix = ShortestSuffix(scanner, std::string_view(rend + 1, rbegin - rend), throughEndMark, throughBeginMark);
	return suffix.data() ? suffix.data() - 1 : nullptr;
}


template<class Scanner>
class RunHelper {
public:
	RunHelper(const Scanner& sc, typename Scanner::State st): Sc(&sc), St(st) {}
	explicit RunHelper(const Scanner& sc): Sc(&sc) { Sc->Initialize(St); }

	RunHelper<Scanner>& Step(Char letter) { Pire::Step(*Sc, St, letter); return *this; }
	RunHelper<Scanner>& Run(TStringBuf str) { Pire::Run(*Sc, St, str); return *this; }
	RunHelper<Scanner>& Run(const char* begin, const char* end) { return Run(TStringBuf(begin, end)); }
	RunHelper<Scanner>& Run(const char* begin, size_t size) { return Run(TStringBuf(begin, begin + size)); }
	RunHelper<Scanner>& Begin() { return Step(BeginMark); }
	RunHelper<Scanner>& End() { return Step(EndMark); }

	const typename Scanner::State& State() const { return St; }
	struct Tag {};
	operator const Tag*() const { return Sc->Final(St) ? (const Tag*) this : 0; }
	bool operator ! () const { return !Sc->Final(St); }

private:
	const Scanner* Sc;
	typename Scanner::State St;
};

template<class Scanner>
RunHelper<Scanner> Runner(const Scanner& sc) { return RunHelper<Scanner>(sc); }

template<class Scanner>
RunHelper<Scanner> Runner(const Scanner& sc, typename Scanner::State st) { return RunHelper<Scanner>(sc, st); }


/// Provided for testing purposes and convinience
template<class Scanner>
bool Matches(const Scanner& scanner, TStringBuf str)
{
	return Runner(scanner).Run(str);
}

template<class Scanner>
bool Matches(const Scanner& scanner, const char* begin, const char* end)
{
	return Runner(scanner).Run(TStringBuf(begin, end));
}

/// Constructs an inline scanner in one statement
template<class Scanner>
Scanner MmappedScanner(const char* ptr, size_t size)
{
	Scanner s;
	s.Mmap(ptr, size);
	return s;
}

}

#endif
