/*
 * encoding.cpp -- implementation of the encodings shipped with Pire.
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


#include <stdexcept>
#include <util/charset/utf8.h>
#include <utility>

#include <contrib/libs/pire/pire/stub/defaults.h>
#include <contrib/libs/pire/pire/stub/utf8.h>
#include <contrib/libs/pire/pire/stub/singleton.h>

#include "encoding.h"
#include "fsm.h"


namespace Pire {

namespace {

	class Latin1: public Encoding {
	public:
		Latin1() : Encoding() {}

		wchar32 FromLocal(const char*& begin, const char* end) const
		{
			if (begin == end)
				throw Error("EOF reached in Pire::Latin1::fromLocal()");
			else if (static_cast<unsigned char>(*begin) >= 0x80)
				throw Error("Pire::Latin1::fromLocal(): wrong character encountered (>=0x80)");
			else
				return (wchar32) *begin++;
		}

		ystring ToLocal(wchar32 ch) const
		{
			if (ch < 0x80)
				return ystring(1, (char) ch);
			else
				return ystring();
		}

		void AppendDot(Fsm& fsm) const { fsm.AppendDot(); }
	};

	namespace UtfRanges {

		static const size_t MaxLen = 4;
		static const size_t First[MaxLen][2] = {
			{0x00, 0x80},
			{0xC0, 0xE0},
			{0xE0, 0xF0},
			{0xF0, 0xF8}
		};
		static const size_t Next[2] = {0x80, 0xC0};
	}


	class Utf8: public Encoding {
	public:
		Utf8() : Encoding() {}

		wchar32 FromLocal(const char*& begin, const char* end) const
		{
			wchar32 rune;
			size_t len;
			if (SafeReadUTF8Char(rune, len, reinterpret_cast<const unsigned char*>(begin), reinterpret_cast<const unsigned char*>(end)) != RECODE_OK)
				throw Error("Error reading UTF8 sequence");
			begin += len;
			return rune;
		}

		ystring ToLocal(wchar32 c) const
		{
			ystring ret(UTF8RuneLenByUCS(c), ' ');
			size_t len;
			unsigned char* p = (unsigned char*) &*ret.begin();
			if (SafeWriteUTF8Char(c, len, p, p + ret.size()) != RECODE_OK)
				Y_ASSERT(!"Pire::UTF8::toLocal(): Internal error");
			return ret;
		}

		void AppendDot(Fsm& fsm) const
		{
			size_t last = fsm.Resize(fsm.Size() + UtfRanges::MaxLen);
			for (size_t i = 0; i < UtfRanges::MaxLen; ++i)
				for (size_t letter = UtfRanges::First[i][0]; letter < UtfRanges::First[i][1]; ++letter)
					fsm.ConnectFinal(fsm.Size() - i - 1, letter);
			for (size_t i = 0; i < UtfRanges::MaxLen - 1; ++i)
				for (size_t letter = UtfRanges::Next[0]; letter < UtfRanges::Next[1]; ++letter)
					fsm.Connect(last + i, last + i + 1, letter);
			fsm.ClearFinal();
			fsm.SetFinal(fsm.Size() - 1, true);
			fsm.SetIsDetermined(false);
		}
	};
}

namespace Encodings {

	const Encoding& Utf8()
	{
		static const Pire::Utf8 utf8;
		return utf8;
	}

	const Encoding& Latin1()
	{
		static const Pire::Latin1 latin1;
		return latin1;
	}

}

}
