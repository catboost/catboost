#ifndef PIRE_MINIMIZE_H
#define PIRE_MINIMIZE_H

#include "stub/stl.h"
#include "partition.h"

namespace Pire {
	namespace Impl {

		/**
		 * An interface of a minimization task.
		 * You don't have to derive from this class; it is just a start point template.
		 */
		class MinimizeTask {
		private:
			struct ImplementationSpecific1;

		public:
			// States must be represented by size_t.

			/// States must be initially divided into some equivalence classes.
			/// If states are in the same equivalence class, they may be merged without loosing state specific info.
			/// Equivalence classes must have indexes from 0 to (Classes - 1).
			/// The algorithm will modify equivalent classes and in the end
			/// all states in the same equivalent class can be merged into one state
			TVector<size_t>& GetStateClass() { return StateClass; }

			/// Returns number of equivalent classes
			size_t& GetClassesNumber() { return Classes; }

			/// Should return number of letter classes
			size_t LettersCount() const;

			/// Should return true if FSM is determined.
			bool IsDetermined() const;

			/// Should return number of states.
			size_t Size() const;

			/// Should calculate vector of previous states by, given the current state and incoming letter class index.
			const TVector<size_t>& Previous(size_t state, size_t letter) const;

			/// Called when states equivalent classes are formed, and written in StateClass.
			void AcceptStates();

			typedef bool Result;

			Result Success() { return true; }

			Result Failure() { return false; }

		private:
			TVector<size_t> StateClass;

			size_t Classes;
		};

		// Minimizes Determined FSM using Hopcroft algorithm, works in O(Size * log(Size) * MaxChar) time,
		// requires O(Size * MaxChar * sizof(size_t)) memory.
		template<class Task>
		typename Task::Result Minimize(Task& task)
		{
			// Minimization algorithm is only applicable to a determined FSM.
			if (!task.IsDetermined()) {
				return task.Failure();
			}

			typedef ypair<size_t, size_t> ClassLetter;

			TVector<ybitset<MaxChar>> queuedClasses(task.Size());

			TDeque<ClassLetter> classesToProcess;

			TVector<TVector<size_t>> classStates(task.Size());

			TVector<size_t>& stateClass = task.GetStateClass();

			for (size_t state = 0; state < task.Size(); ++state) {
				classStates[stateClass[state]].push_back(state);
			}

			for (size_t classIndex = 0; classIndex < task.GetClassesNumber(); ++classIndex) {
				for (size_t letter = 0; letter < task.LettersCount(); ++letter) {
					classesToProcess.push_back(ymake_pair(classIndex, letter));
					queuedClasses[classIndex][letter] = 1;
				}
			}

			TVector<size_t> classChange(task.Size());
			TVector<TVector<size_t>> removedStates(task.Size());

			while (classesToProcess.size()) {
				const auto currentClass = classesToProcess.front().first;
				const auto currentLetter = classesToProcess.front().second;
				classesToProcess.pop_front();
				queuedClasses[currentClass][currentLetter] = 0;
				TVector<size_t> splittedClasses;

				for (const auto& classState : classStates[currentClass]) {
					for (const auto& state: task.Previous(classState, currentLetter)) {
						if (classChange[stateClass[state]] != task.GetClassesNumber()) {
							classChange[stateClass[state]] = task.GetClassesNumber();
							splittedClasses.push_back(stateClass[state]);
						}
						removedStates[stateClass[state]].push_back(state);
					}
				}


				for (const auto& splittedClass : splittedClasses) {
					if (removedStates[splittedClass].size() == classStates[splittedClass].size()) {
						classChange[splittedClass] = 0;
						removedStates[splittedClass].clear();
						continue;
					}

					const auto newClass = task.GetClassesNumber()++;
					classChange[splittedClass] = newClass;
					std::swap(classStates[newClass], removedStates[splittedClass]);
					for (const auto& state : classStates[newClass]) {
						stateClass[state] = newClass;
					}

					auto iter = classStates[splittedClass].begin();
					for (const auto state : classStates[splittedClass]) {
						if (stateClass[state] == splittedClass) {
							*iter = state;
							++iter;
						}
					}
					classStates[splittedClass].erase(iter, classStates[splittedClass].end());

					for (size_t letter = 0; letter < task.LettersCount(); ++letter) {
						if (queuedClasses[splittedClass][letter]
							|| classStates[splittedClass].size() > classStates[newClass].size()) {

							queuedClasses[newClass][letter] = 1;
							classesToProcess.push_back(ymake_pair(newClass, letter));
						} else {
							queuedClasses[splittedClass][letter] = 1;
							classesToProcess.push_back(ymake_pair(splittedClass, letter));
						}
					}
				}
			}

			task.AcceptStates();
			return task.Success();
		}
	}
}

#endif
