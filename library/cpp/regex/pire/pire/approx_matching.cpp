/*
 * approx_matching.cpp -- implementation of CreateApproxFsm function
 *
 * Copyright (c) 2019 YANDEX LLC, Karina Usmanova <usmanova.karin@yandex.ru>
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


#include "approx_matching.h"

namespace Pire {
	Fsm CreateApproxFsm(const Fsm& regexp, size_t distance) {
		Fsm approxFsm = regexp;

		TVector<TSet<Char>> outgoingLettersTable(regexp.Size());
		for (size_t state = 0; state < regexp.Size(); ++state) {
			outgoingLettersTable[state] = regexp.OutgoingLetters(state);
		}

		TVector<TMap<Char, Fsm::StatesSet>> destinationsTable(regexp.Size());
		for (size_t state = 0; state < regexp.Size(); ++state) {
			for (Char letter : outgoingLettersTable[state]) {
				destinationsTable[state][letter] = regexp.Destinations(state, letter);
			}
		}

		for (size_t fsmIdx = 0; fsmIdx < distance; ++fsmIdx) {
			approxFsm.Import(regexp);
			const auto shift = fsmIdx * regexp.Size();

			for (size_t state = 0; state < regexp.Size(); ++state) {
				for (Char letter : outgoingLettersTable[state]) {
					for (size_t to : destinationsTable[state][letter]) {
						for (Char ch = 0; ch < MaxChar; ++ch) {
							if (!approxFsm.Connected(state + shift, to + shift, ch)) {
								approxFsm.Connect(state + shift, to + shift + regexp.Size(), ch);
							}
						}

						approxFsm.Connect(state + shift, to + shift + regexp.Size(), Epsilon);
					}

					for (Char ch = 0; ch < MaxChar; ++ch) {
						approxFsm.Connect(state + shift, state + shift + regexp.Size(), ch);
					}
				}

				if (regexp.IsFinal(state)) {
					approxFsm.SetFinal(state + shift + regexp.Size(), true);
				}
			}
		}

		size_t maxState = (distance > 0) ? approxFsm.Size() - regexp.Size() : 0;
		for (size_t state = 0; state < maxState; ++state) {
			size_t currentDist = state / regexp.Size();
			size_t intState = state % regexp.Size();

			for (Char firstLetter : outgoingLettersTable[intState]) {
				for (size_t firstDest : destinationsTable[intState][firstLetter]) {
					for (Char secondLetter : outgoingLettersTable[firstDest]) {
						for (size_t secondDest : destinationsTable[firstDest][secondLetter]) {
							if (secondDest != intState || firstDest != intState) {
								approxFsm.Resize(approxFsm.Size() + 1);

								size_t to = secondDest + (currentDist + 1) * regexp.Size();
								size_t middle = approxFsm.Size() - 1;

								approxFsm.Connect(state, middle, secondLetter);
								approxFsm.Connect(middle, to, firstLetter);
							}
						}
					}
				}
			}
		}

		return approxFsm;
	}
}
