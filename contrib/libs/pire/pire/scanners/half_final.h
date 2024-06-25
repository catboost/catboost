#ifndef PIRE_SCANNERS_HALF_FINAL_H
#define PIRE_SCANNERS_HALF_FINAL_H

#include <string.h>
#include "common.h"
#include "multi.h"
#include <contrib/libs/pire/pire/fsm.h>
#include <contrib/libs/pire/pire//half_final_fsm.h>
#include <contrib/libs/pire/pire//stub/stl.h>

namespace Pire {

namespace Impl {


/*
 * A half final scanner -- the deterministic scanner having half-terminal states,
 * so it matches regexps in all terminal transitional states.
 *
 * The scanner can also count the number of substrings, that match each regexp. These substrings may intersect.
 *
 * Comparing it with scanner, it runs slower, but allows to glue significantly
 * larger number of scanners into one within the same size limits.
 *
 * The class is subclass of Scanner, having the same methods, but different state type.
 *
 * There are no restrictions for regexps and fsm's, for which it is built, but it
 * does not work properly if the matching text does not end with EndMark.
 *
 * For count to work correctly, the fsm should not be determined.
 */
template<typename Relocation, typename Shortcutting>
class HalfFinalScanner : public Scanner<Relocation, Shortcutting> {
public:
	typedef typename Impl::Scanner<Relocation, Shortcutting> Scanner;

	HalfFinalScanner() : Scanner() {}

	explicit HalfFinalScanner(Fsm fsm_, size_t distance = 0) {
		if (distance) {
			fsm_ = CreateApproxFsm(fsm_, distance);
		}
		HalfFinalFsm fsm(fsm_);
		fsm.MakeScanner();
		Scanner::Init(fsm.GetFsm().Size(), fsm.GetFsm().Letters(), fsm.GetFsm().Finals().size(), fsm.GetFsm().Initial(), 1);
		BuildScanner(fsm.GetFsm(), *this);
	}

	explicit HalfFinalScanner(const HalfFinalFsm& fsm) {
		Scanner::Init(fsm.GetFsm().Size(), fsm.GetFsm().Letters(), fsm.GetTotalCount(), fsm.GetFsm().Initial(), 1);
		BuildScanner(fsm.GetFsm(), *this);
		BuildFinals(fsm);
	}

	typedef typename Scanner::ScannerRowHeader ScannerRowHeader;
	typedef typename Scanner::Action Action;

	class State {
	public:
		typedef TVector<size_t>::const_iterator IdsIterator;

		State() : ScannerState(0) {}

		State(const typename Scanner::State& otherState) : ScannerState(otherState) {}

		void GetMatchedRegexpsIds() {
			MatchedRegexpsIds.clear();
			for (size_t i = 0; i < MatchedRegexps.size(); i++) {
				if (MatchedRegexps[i]) {
					MatchedRegexpsIds.push_back(i);
				}
			}
		}

		IdsIterator IdsBegin() const {
			return MatchedRegexpsIds.cbegin();
		}

		IdsIterator IdsEnd() const {
			return MatchedRegexpsIds.cend();
		}

		bool operator==(const State& other) const {
			return ScannerState == other.ScannerState && MatchedRegexps == other.MatchedRegexps;
		}

		bool operator!=(const State& other) const {
			return ScannerState != other.ScannerState || MatchedRegexps != other.MatchedRegexps;
		}

		size_t Result(size_t regexp_id) const {
			return MatchedRegexps[regexp_id];
		}

		void Save(yostream* s) const {
			SavePodType(s, Pire::Header(5, sizeof(size_t)));
			Impl::AlignSave(s, sizeof(Pire::Header));
			auto stateSizePair = ymake_pair(ScannerState, MatchedRegexps.size());
			SavePodType(s, stateSizePair);
			Impl::AlignSave(s, sizeof(ypair<size_t, size_t>));
			Y_ASSERT(0);
		}

		void Load(yistream* s) {
			Impl::ValidateHeader(s, 5, sizeof(size_t));
			ypair<size_t, size_t> stateSizePair;
			LoadPodType(s, stateSizePair);
			Impl::AlignLoad(s, sizeof(ypair<size_t, size_t>));
			ScannerState = stateSizePair.first;
			MatchedRegexps.clear();
			MatchedRegexps.resize(stateSizePair.second);
		}

	private:
		TVector<size_t> MatchedRegexpsIds;
		typename Scanner::State ScannerState;
		TVector<size_t> MatchedRegexps;

		friend class HalfFinalScanner<Relocation, Shortcutting>;
	};


	/// Checks whether specified state is in any of the final sets
	bool Final(const State& state) const { return Scanner::Final(state.ScannerState); }

	/// Checks whether specified state is 'dead' (i.e. scanner will never
	/// reach any final state from current one)
	bool Dead(const State& state) const { return Scanner::Dead(state.ScannerState); }

	typedef ypair<typename State::IdsIterator, typename State::IdsIterator> AcceptedRegexpsType;

	AcceptedRegexpsType AcceptedRegexps(State& state) const {
		state.GetMatchedRegexpsIds();
		return ymake_pair(state.IdsBegin(), state.IdsEnd());
	}

	/// Returns an initial state for this scanner
	void Initialize(State& state) const {
		state.ScannerState = Scanner::m.initial;
		state.MatchedRegexps.clear();
		state.MatchedRegexps.resize(Scanner::m.regexpsCount);
		TakeAction(state, 0);
	}

	Action NextTranslated(State& state, Char letter) const {
		return Scanner::NextTranslated(state.ScannerState, letter);
	}

	/// Handles one character
	Action Next(State& state, Char c) const {
		return Scanner::NextTranslated(state.ScannerState, Scanner::Translate(c));
	}

	void TakeAction(State& state, Action) const {
		if (Final(state)) {
			size_t idx = StateIndex(state);
			const size_t *it = Scanner::m_final + Scanner::m_finalIndex[idx];
			while (*it != Scanner::End) {
				state.MatchedRegexps[*it]++;
				++it;
			}
		}
	}

	HalfFinalScanner(const HalfFinalScanner& s) : Scanner(s) {}

	HalfFinalScanner(const Scanner& s) : Scanner(s) {}

	HalfFinalScanner(HalfFinalScanner&& s) : Scanner(s) {}

	HalfFinalScanner(Scanner&& s) : Scanner(s) {}

	template<class AnotherRelocation>
	HalfFinalScanner(const HalfFinalScanner<AnotherRelocation, Shortcutting>& s)
			: Scanner(s) {}

	template<class AnotherRelocation>
	HalfFinalScanner(const Impl::Scanner<AnotherRelocation, Shortcutting>& s) : Scanner(s) {}

	void Swap(HalfFinalScanner& s) {
		Scanner::Swap(s);
	}

	HalfFinalScanner& operator=(const HalfFinalScanner& s) {
		HalfFinalScanner(s).Swap(*this);
		return *this;
	}

	size_t StateIndex(const State& s) const {
		return Scanner::StateIndex(s.ScannerState);
	}

	/**
	 * Agglutinates two scanners together, producing a larger scanner.
	 * Checking a string against that scanner effectively checks them against both agglutinated regexps
	 * (detailed information about matched regexps can be obtained with AcceptedRegexps()).
	 *
	 * Returns default-constructed scanner in case of failure
	 * (consult Scanner::Empty() to find out whether the operation was successful).
	 */
	static HalfFinalScanner Glue(const HalfFinalScanner& a, const HalfFinalScanner& b, size_t maxSize = 0) {
		return Scanner::Glue(a, b, maxSize);
	}

	ScannerRowHeader& Header(const State& s) { return Scanner::Header(s.ScannerState); }

	const ScannerRowHeader& Header(const State& s) const { return Scanner::Header(s.ScannerState); }

private:
	void BuildFinals(const HalfFinalFsm& fsm) {
		Y_ASSERT(Scanner::m_buffer);
		Y_ASSERT(fsm.GetFsm().Size() == Scanner::Size());
		auto finalWriter = Scanner::m_final;
		for (size_t state = 0; state < Scanner::Size(); ++state) {
			Scanner::m_finalIndex[state] = finalWriter - Scanner::m_final;
			for (size_t i = 0; i < fsm.GetCount(state); i++) {
				*finalWriter++ = 0;
			}
			*finalWriter++ = static_cast<size_t>(-1);
		}
	}

	template<class Scanner>
	friend void Pire::BuildScanner(const Fsm&, Scanner&);

	typedef State InternalState; // Needed for agglutination
};

}


typedef Impl::HalfFinalScanner<Impl::Relocatable, Impl::ExitMasks<2> > HalfFinalScanner;
typedef Impl::HalfFinalScanner<Impl::Relocatable, Impl::NoShortcuts> HalfFinalScannerNoMask;

/**
 * Same as above, but does not allow relocation or mmap()-ing.
 * On the other hand, runs faster than HalfFinal.
 */
typedef Impl::HalfFinalScanner<Impl::Nonrelocatable, Impl::ExitMasks<2> > NonrelocHalfFinalScanner;
typedef Impl::HalfFinalScanner<Impl::Nonrelocatable, Impl::NoShortcuts> NonrelocHalfFinalScannerNoMask;

}


namespace std {
	inline void swap(Pire::HalfFinalScanner& a, Pire::HalfFinalScanner& b) {
		a.Swap(b);
	}

	inline void swap(Pire::NonrelocHalfFinalScanner& a, Pire::NonrelocHalfFinalScanner& b) {
		a.Swap(b);
	}
}

#endif
