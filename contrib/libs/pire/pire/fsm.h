/*
 * fsm.h -- the definition of the FSM class.
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


#ifndef PIRE_FSM_H
#define PIRE_FSM_H


#include <contrib/libs/pire/pire/stub/stl.h>

#include "partition.h"
#include "defs.h"

namespace Pire {

	namespace Impl {
		class FsmDetermineTask;
		class FsmMinimizeTask;
        class HalfFinalDetermineTask;
	}

	/// A Flying Spaghetti Monster... no, just a Finite State Machine.
	class Fsm {
	public:		
		typedef ybitset<MaxChar> Charset;

		Fsm();
		void Swap(Fsm& fsm);

		static Fsm MakeFalse();

		/// Current number of states
		size_t Size() const { return m_transitions.size(); }

		Fsm& Append(char c);
		Fsm& Append(const ystring& str);
		Fsm& AppendSpecial(Char c);

		/// Efficiently appends a union of passed strings to FSM.
		/// Used for ranges (e.g. [a-z]), character classes (e.g. \w, \d)
		/// and case-insensitive comparison of multibyte characters,
		/// when one string represents a lowercase variant of a character,
		/// while another string represents its uppercase variant.
		Fsm& AppendStrings(const TVector<ystring>& strings);

		/// Appends a part matching a single byte (any).
		Fsm& AppendDot();

		/// Appends and prepends the FSM with the iterated dot (see above).
		Fsm& Surround(); // returns *this
		Fsm Surrounded() const { Fsm copy(*this); copy.Surround(); return copy; }

		Fsm& operator += (const Fsm& rhs); ///< Concatenation
		Fsm& operator |= (const Fsm& rhs); ///< Alternation
		Fsm& operator &= (const Fsm& rhs); ///< Conjunction
		Fsm& Iterate();                    ///< Klene star
		Fsm& Complement();                 ///< Complementation
		Fsm& operator *= (size_t count) { *this = *this * count; return *this; }

		Fsm operator + (const Fsm& rhs) const { Fsm a(*this); return a += rhs; }
		Fsm operator | (const Fsm& rhs) const { Fsm a(*this); return a |= rhs; }
		Fsm operator & (const Fsm& rhs) const { Fsm a(*this); return a &= rhs; }
		Fsm operator * ()               const { Fsm a(*this); return a.Iterate(); }
		Fsm operator ~ ()               const { Fsm a(*this); return a.Complement(); }
		Fsm operator * (size_t count) const;

		// === Raw FSM construction ===
		
		/// Connects two states with given transition
		void Connect(size_t from, size_t to, Char c = Epsilon);
		
		/// Removes given character from the specified transition.
		void Disconnect(size_t from, size_t to, Char c);
		
		/// Completely removes given transition
		void Disconnect(size_t from, size_t to);

		/// Creates an FSM which matches any prefix of any word current FSM matches.
		void MakePrefix();

		/// Creates an FSM which matches any suffix of any word current FSM matches.
		void MakeSuffix();
		
		/// Does the one way part of Surround().
		void PrependAnything();
		void AppendAnything();
		
		/// Creates an FSM which matches reversed strings matched by current FSM.
		Fsm& Reverse();

		/// Returns a set of states from which no final states are reachable
		TSet<size_t> DeadStates() const;

		/// Removes all dead end paths from FSM
		void RemoveDeadEnds();

		/// Determines and minimizes the FSM if neccessary. Returns *this.
		Fsm& Canonize(size_t maxSize = 0);

		template<class Scanner>
		Scanner Compile(size_t distance = 0);

		void DumpState(yostream& s, size_t state) const;
		void DumpTo(yostream& s, const ystring& name = "") const;

		typedef TSet<size_t> StatesSet;
		typedef TMap<size_t, StatesSet> TransitionRow;
		typedef TVector<TransitionRow> TransitionTable;

		struct LettersEquality {
			LettersEquality(const Fsm::TransitionTable& tbl): m_tbl(&tbl) {}
			bool operator()(Char a, Char b) const;
		private:
			const Fsm::TransitionTable* m_tbl;
		};

		typedef TSet<size_t> FinalTable;
		typedef Partition<Char, LettersEquality> LettersTbl;


		/*
		 * A very low level FSM building interface.
		 *
		 * It is generally unwise to call any of these functions unless you are building
		 * your own scanner, your own ecoding or exaclty know what you are doing.
		 */
		unsigned long Tag(size_t state) const { Tags::const_iterator i = tags.find(state); return (i == tags.end()) ? 0 : i->second; }
		void SetTag(size_t state, unsigned long tag) { tags[state] = tag; }

		unsigned long Output(size_t from, size_t to) const;
		void SetOutput(size_t from, size_t to, unsigned long output) { outputs[from][to] = output; }
		void ClearOutputs() { outputs.clear(); }

		const FinalTable& Finals() const { return m_final; }
		bool IsFinal(size_t state) const { return m_final.find(state) != m_final.end(); }
		void SetFinal(size_t size, bool final);
		void ClearFinal() { m_final.clear(); }

		/// Removes all espilon transitions from the FSM. Does not change the FSMs language.
		void RemoveEpsilons();

		/// Resize FSM to newSize states. Returns old size.
		size_t Resize(size_t newSize);

		/// Imports foreign transition table
		void Import(const Fsm& rhs);

		/// Connects all final state with given state
		void ConnectFinal(size_t to, Char c = Epsilon);

		/// Diverts all transition between two given states to @p dest, preserving outputs
		void Divert(size_t from, size_t to, size_t dest);

		/// Checks whether two states are connected using given letter.
		bool Connected(size_t from, size_t to, Char c) const;
		
		/// Returns a set of letters on which a transition from the specified state exists
		TSet<Char> OutgoingLetters(size_t state) const;
		
		/// Returns a set of states where a transition from the given state using the given letter is possible
		const StatesSet& Destinations(size_t from, Char letter) const;

		/// Checks whether two states are connected using any letter.
		bool Connected(size_t from, size_t to) const;
		size_t Initial() const { return initial; }
		void SetInitial(size_t init) { initial = init; }
		
		const LettersTbl& Letters() const { return letters; }
		
		/// Determines the FSM.
		/// Breaks FSM invariant of having a single final state, so high-level FSM building
		/// functions (i.e. Append(), operator+(), etc...) no longer can be applied to the FSM
		/// until the invariants have been manually restored.
		/// return value: successful?
		bool Determine(size_t maxsize = 0);
		bool IsDetermined() const { return determined; }
		void SetIsDetermined(bool det) { determined = det; }

		/// Minimizes amount of states in the regexp.
		/// Requires a determined FSM.
		void Minimize();


		/// Builds letters equivalence classes
		void Sparse(bool needEpsilons = false);

		/// Unpacks all letters equivalence classs back into transitions table
		void Unsparse();

	private:

		/// Transitions table :: Q x V -> exp(Q)
		TransitionTable m_transitions;

		/// Initial state
		size_t initial;

		/// Final states.
		FinalTable m_final;

		LettersTbl letters;
		
		/// Does 'letters' make sense?
		bool m_sparsed;

		/// Is the FSM already determined?
		bool determined;

		/// Output
		typedef TMap< size_t, TMap<size_t, unsigned long> > Outputs;
		Outputs outputs;
		
		typedef TMap<size_t, unsigned long> Tags;
		Tags tags;

		/// Heuristics hit: true iff this FSM is a union of two other FSMs
		bool isAlternative;
		
		void ShortCutEpsilon(size_t from, size_t thru, TVector< TSet<size_t> >& inveps); ///< internal
		void MergeEpsilonConnection(size_t from, size_t to); ///< internal

		TSet<size_t> TerminalStates() const;
		
		Char Translate(Char c) const;
		
		void ClearHints() { isAlternative = false; }
		
		friend class Impl::FsmDetermineTask;
		friend class Impl::FsmMinimizeTask;
        friend class Impl::HalfFinalDetermineTask;
	};

	template<class Scanner>
	void BuildScanner(const Fsm& fsm, Scanner& r)
	{
		TSet<size_t> dead;
		if (Scanner::DeadFlag)
			dead = fsm.DeadStates();

		for (size_t state = 0; state < fsm.Size(); ++state)
			r.SetTag(state, typename Scanner::Tag(fsm.Tag(state)
				| (fsm.IsFinal(state) ? Scanner::FinalFlag : 0)
				| ((dead.find(state) != dead.end()) ? Scanner::DeadFlag : 0)));

		for (size_t from = 0; from != fsm.Size(); ++from)
			for (Fsm::LettersTbl::ConstIterator lit = fsm.Letters().Begin(), lie = fsm.Letters().End(); lit != lie; ++lit) {
				const Fsm::StatesSet& tos = fsm.Destinations(from, lit->first);
				for (Fsm::StatesSet::const_iterator to = tos.begin(), toEnd = tos.end(); to != toEnd; ++to)
					r.SetJump(from, lit->first, *to, r.RemapAction(fsm.Output(from, *to)));
			}
		
		r.FinishBuild();
	}

	template<class Scanner>
	inline Scanner Fsm::Compile(size_t distance)
	{
		return Scanner(*this, distance);
	}

	yostream& operator << (yostream&, const Fsm&);
}

#endif
