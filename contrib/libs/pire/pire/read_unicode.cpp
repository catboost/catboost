/*
 * read_unicode.cpp -- implementation of the UnicodeReader.
 *
 * Copyright (c) 2019 YANDEX LLC
 * Author: Karina Usmanova <usmanova.karin@yandex.ru>
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


#include "read_unicode.h"

#include <contrib/libs/pire/pire/re_lexer.h>

namespace Pire {
	wchar32 UnicodeReader::ReadUnicodeCharacter() {
		ystring hexStr;
		GetChar();
		wchar32 ch = PeekChar();

		if (ch == '{') {
			GetChar();
			hexStr = ReadHexDigit(
					[](wchar32 ch, size_t numAdded) -> bool { return ch == End || (numAdded != 0 && ch == '}'); });
			ch = GetChar();
			if (ch != '}') {
				Error("Pire::UnicodeReader::ReadUnicodeCharacter(): \"\\x{...\" sequence should be closed by \"}\"");
			}
		} else {
			hexStr = ReadHexDigit([](wchar32, size_t numAdded) -> bool { return numAdded == 2; });
			if (hexStr.size() != 2) {
				Error("Pire::UnicodeReader::ReadUnicodeCharacter(): \"\\x...\" sequence should contain two symbols");
			}
		}
		return HexToDec(hexStr);
	}

	bool UnicodeReader::IsHexDigit(wchar32 ch) {
		return ch < 256 && std::isxdigit(ch) != 0;
	}

	ystring UnicodeReader::ReadHexDigit(std::function<bool(wchar32, size_t)> shouldStop) {
		ystring result;
		wchar32 ch = GetChar();
		while (!shouldStop(ch, result.size())) {
			if (!IsHexDigit(ch)) {
				Error("Pire::UnicodeReader::ReadHexDigit(): \"\\x...\" sequence contains non-valid hex number");
			}
			result.push_back(ch);
			ch = GetChar();
		}
		UngetChar(ch);
		return result;
	}

	wchar32 UnicodeReader::HexToDec(const ystring &hexStr) {
		wchar32 converted;
		try {
			converted = std::stoul(hexStr, 0, 16);
		} catch (std::out_of_range &) {
			converted = MAX_UNICODE + 1;
		}
		if (converted > MAX_UNICODE) {
			Error("Pire::UnicodeReader::HexToDec(): hex number in \"\\x...\" sequence is too large");
		}
		return converted;
	}
}


