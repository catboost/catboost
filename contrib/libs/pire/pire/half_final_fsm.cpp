#include "defs.h"
#include "determine.h"
#include "half_final_fsm.h"

namespace Pire {
	size_t HalfFinalFsm::MaxCountDepth = 10;

	void HalfFinalFsm::MakeScanner() {
		fsm.Canonize();
		bool allowHalfFinals = AllowHalfFinals();
		if (!allowHalfFinals) {
			MakeHalfFinal();
			return;
		}
		DisconnectFinals(true);
	}

	bool HalfFinalFsm::AllowHalfFinals() {
		fsm.Canonize();
		for (size_t state = 0; state < fsm.Size(); ++state) {
			if (fsm.IsFinal(state)) {
				for (const auto& let : fsm.Letters()) {
					bool hasFinalTransition = fsm.Destinations(state, let.first).empty();
					for (const auto& to : fsm.Destinations(state, let.first)) {
						if (fsm.IsFinal(to)) {
							hasFinalTransition = true;
						}
					}
					if (!hasFinalTransition) {
						return false;
					}
				}
			}
		}
		return true;
	}

	void HalfFinalFsm::MakeHalfFinal() {
		fsm.Unsparse();
		const auto newFinal = fsm.Size();
		fsm.Resize(newFinal + 1);
		for (unsigned letter = 0; letter < MaxChar; ++letter) {
			if (letter != Epsilon) {
				fsm.Connect(newFinal, newFinal, letter);
			}
		}
		for (size_t state = 0; state < fsm.Size(); ++state) {
			bool hasFinalTransitions = false;
			for (const auto& to : fsm.Destinations(state, EndMark)) {
				if (fsm.IsFinal(to)) {
					hasFinalTransitions = true;
					break;
				}
			}
			if (hasFinalTransitions) {
				Fsm::StatesSet destinations = fsm.Destinations(state, EndMark);
				for (const auto& to : destinations) {
					fsm.Disconnect(state, to, EndMark);
				}
				fsm.Connect(state, newFinal, EndMark);
			}
		}
		fsm.ClearFinal();
		fsm.SetFinal(newFinal, true);
		fsm.Sparse();
	}

	void HalfFinalFsm::DisconnectFinals(bool allowIntersects) {
		fsm.Unsparse();
		for (size_t state = 0; state != fsm.Size(); ++state) {
			fsm.SetTag(state, 0);
			if (fsm.IsFinal(state)) {
				for (unsigned letter = 0; letter < MaxChar; ++letter) {
					Fsm::StatesSet destinations = fsm.Destinations(state, letter);
					for (const auto& to : destinations) {
						fsm.Disconnect(state, to, letter);
					}
				}
				if (!allowIntersects) {
					fsm.Connect(state, fsm.Initial());
				}
			}
		}
		if (allowIntersects) {
			fsm.PrependAnything();
		}
		fsm.Sparse();
		fsm.SetIsDetermined(false);
		fsm.Canonize();
	}

	void HalfFinalFsm::MakeNonGreedyCounter(bool allowIntersects /* = true */, bool simplify /* = true */) {
		fsm.Canonize();
		fsm.PrependAnything();
		fsm.RemoveDeadEnds();
		fsm.Canonize();
		if (!allowIntersects || simplify) {
			DisconnectFinals(allowIntersects);
		}
	}

	void HalfFinalFsm::MakeGreedyCounter(bool simplify /* = true */) {
		fsm.Canonize();
		fsm.RemoveDeadEnds();
		size_t determineFactor = MaxCountDepth;
		if (simplify) {
			determineFactor = 1;
		}
		Determine(determineFactor);
		if (simplify) {
			fsm.Minimize();
		}
		fsm.RemoveDeadEnds();
	}

	namespace Impl {

		class HalfFinalDetermineState {
		public:
			HalfFinalDetermineState(const Fsm& fsm, bool initial = false, size_t lastFinalCount = 0)
				: mFsm(fsm)
				, ToAdd(0)
				, LastFinalCount(lastFinalCount)
			{
				if (initial) {
					FinishBuild(1);
				}
			}

			HalfFinalDetermineState Next(Char letter, size_t maxCount) const {
				HalfFinalDetermineState next(mFsm, false, LastFinalCount);
				for (const auto& state : States) {
					for (const auto& nextState : mFsm.Destinations(state.State, letter)) {
						next.AddState(nextState, state.Count, state.ReachedFinal);
					}
				}
				next.FinishBuild(maxCount, States.back().Count);
				if (letter == EndMark) {
					next.ToAdd += next.LastFinalCount;
					next.LastFinalCount = 0;
					next.States.clear();
					next.AddState(mFsm.Initial(), 0, false, true);
					return next;
				}
				return next;
			}

			void CopyData(Fsm& newFsm, size_t index) const {
				if (ToAdd) {
					newFsm.SetFinal(index, true);
					newFsm.SetTag(index, ToAdd);
				}
			}

			bool operator<(const HalfFinalDetermineState& otherState) const {
				if (ToAdd != otherState.ToAdd) {
					return ToAdd < otherState.ToAdd;
				}
				if (LastFinalCount != otherState.LastFinalCount) {
					return LastFinalCount < otherState.LastFinalCount;
				}
				return States < otherState.States;
			}

			struct StateHolder {
				size_t State;
				size_t Count;
				bool ReachedFinal;

				bool operator<(const StateHolder& other) const {
					if (State != other.State) {
						return State < other.State;
					}
					if (Count != other.Count) {
						return Count < other.Count;
					}
					return ReachedFinal < other.ReachedFinal;
				}
			};

		private:
			const Fsm& mFsm;
			TVector<StateHolder> States;
			size_t ToAdd;
			size_t LastFinalCount;

			void AddState(size_t state, size_t count, bool reachedFinal, bool force = false) {
				size_t newLastFinalCount = LastFinalCount;
				if (mFsm.IsFinal(state) && !reachedFinal) {
					++count;
					reachedFinal = true;
					newLastFinalCount = count;
				}
				for (const auto& addedState : States) {
					if (addedState.State == state) {
						return;
					}
				}
				if (States.empty() || !mFsm.IsFinal(States.back().State) || force) {
					States.push_back({state, count, reachedFinal});
					LastFinalCount = newLastFinalCount;
				}
			}

			void FinishBuild(size_t maxCount, size_t lastCount = 0) {
				if (!States.empty() && mFsm.IsFinal(States.back().State)) {
					lastCount = States.back().Count;
				}
				AddState(mFsm.Initial(), lastCount, false, true);
				LastFinalCount = std::min(LastFinalCount, maxCount);
				size_t minCount = States[0].Count;
				for (auto& state : States) {
					if (state.Count > maxCount) {
						state.Count = maxCount;
					}
					minCount = std::min(state.Count, minCount);
				}
				ToAdd = minCount;
				for (auto& state : States) {
					state.Count -= minCount;
				}
				LastFinalCount -= minCount;
			}
		};

		class HalfFinalDetermineTask {
		public:
			typedef HalfFinalDetermineState State;
			typedef Fsm::LettersTbl LettersTbl;
			typedef TMap<State, size_t> InvStates;

			HalfFinalDetermineTask(const Fsm& fsm, size_t maxCount)
				: mFsm(fsm)
				, MaxCount(maxCount)
			{
				size_t oldSize = mFsm.Size();
				mFsm.Import(fsm);
				mFsm.Unsparse();
				for (size_t state = 0; state < mFsm.Size(); ++state) {
					for (Char letter = 0; letter < MaxChar; ++letter) {
						Fsm::StatesSet destinations = mFsm.Destinations(state, letter);
						for (const auto destination : destinations) {
							size_t newDestination = destination % oldSize;
							if (letter == EndMark) {
								newDestination += oldSize;
							}
							if (destination != newDestination) {
								mFsm.Disconnect(state, destination, letter);
								mFsm.Connect(state, newDestination, letter);
							}
						}
					}
					if (mFsm.Destinations(state, EndMark).size() == 0) {
						mFsm.Connect(state, oldSize + mFsm.Initial(), EndMark);
					}
				}
				mFsm.Sparse();
			}

			const LettersTbl& Letters() const { return mFsm.Letters(); }

			State Initial() const {
				return State(mFsm, true);
			}

			State Next(const State& state, Char letter) const {
				return state.Next(letter, MaxCount);
			}

			bool IsRequired(const State& /*state*/) const { return true; }

			void AcceptStates(const TVector<State>& newStates) {
				mNewFsm.Resize(newStates.size());
				mNewFsm.SetInitial(0);
				mNewFsm.SetIsDetermined(true);
				mNewFsm.letters = Letters();
				mNewFsm.ClearFinal();
				for (size_t i = 0; i < newStates.size(); i++) {
					newStates[i].CopyData(mNewFsm, i);
				}
			}

			void Connect(size_t from, size_t to, Char letter) {
				Y_ASSERT(mNewFsm.Destinations(from, letter).size() == 0);
				mNewFsm.Connect(from, to, letter);
			}

			typedef bool Result;

			Result Success() { return true; }

			Result Failure() { return false; }

			Fsm& Output() { return mNewFsm; }

			void SetMaxCount(size_t maxCount) {
				MaxCount = maxCount;
			}

		private:
			Fsm mFsm;
			size_t MaxCount;
			Fsm mNewFsm;
		};
	}

	void HalfFinalFsm::Determine(size_t depth) {
		static const unsigned MaxSize = 200000;

		Impl::HalfFinalDetermineTask task(fsm, depth);
		if (!Pire::Impl::Determine(task, MaxSize)) {
			task.SetMaxCount(1);
			Pire::Impl::Determine(task, MaxSize);
		}

		task.Output().Swap(fsm);
	}

	size_t HalfFinalFsm::GetCount(size_t state) const {
		if (fsm.IsFinal(state)) {
			if (fsm.Tag(state)) {
				return fsm.Tag(state);
			} else {
				return 1;
			}
		}
		return 0;
	}

	size_t HalfFinalFsm::GetTotalCount() const {
		size_t count = 0;
		for (size_t state = 0; state < fsm.Size(); ++state) {
			count += GetCount(state);
		}
		return count;
	}
}
