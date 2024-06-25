/*
 * encoding.h -- the interface of Encoding.
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


#ifndef PIRE_ENCODING_H
#define PIRE_ENCODING_H


#include <contrib/libs/pire/pire/stub/defaults.h>
#include <contrib/libs/pire/pire/stub/stl.h>

namespace Pire {

class Fsm;

class Encoding {
public:
	virtual ~Encoding() {}

	/// Should read bytes from @p begin and return the corresponding Unicode
	/// character, advancing @p begin.
	virtual wchar32 FromLocal(const char*& begin, const char* end) const = 0;

	/// Opposite to FromLocal(), transforms given Unicode character into
	/// the string in the encoding.
	virtual ystring ToLocal(wchar32 c) const = 0;

	/// Given the FSM, should append the representation of a dot in the ecoding
	/// to that FSM.
	virtual void AppendDot(Fsm&) const = 0;

	template<class OutputIter>
	OutputIter FromLocal(const char* begin, const char* end, OutputIter iter) const
	{
		while (begin != end) {
			*iter = FromLocal(begin, end);
			++iter;
		}
		return iter;
	}
};

namespace Encodings {
	const Encoding& Latin1();
	const Encoding& Utf8();

};


};

#endif
