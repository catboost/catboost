/*
 * re_lexer.h -- definition required for parsing regexps
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


#ifndef PIRE_RE_LEXER_H
#define PIRE_RE_LEXER_H


#include <vector>
#include <stack>
#include <set>
#include <utility>
#include <stdexcept>
#include <utility>
#include <string.h>
#include <contrib/libs/pire/pire/stub/defaults.h>
#include <contrib/libs/pire/pire/stub/stl.h>

#include "encoding.h"
#include "any.h"

namespace Pire {

namespace Consts {
enum { Inf = -1 };

static const wchar32 Control     = 0xF0000000;
static const wchar32 ControlMask = 0xFF000000;
static const wchar32 End         = Control | 0xFF;
};

using namespace Consts;

namespace TokenTypes {
enum {
	None = 0,
	Letters,
	Count,
	Dot,
	Open,
	Close,
	Or,
	And,
	Not,
	BeginMark,
	EndMark,
	End
};
}

/**
* A single terminal character in regexp pattern.
* Consists of a type (a character, a repetition count, an opening parenthesis, etc...)
* and optional value.
*/
class Term {
public:
	typedef TVector<wchar32> String;
	typedef TSet<String> Strings;

	typedef ypair<int, int> RepetitionCount;
	typedef ypair<Strings, bool> CharacterRange;

	struct DotTag {};
	struct BeginTag {};
	struct EndTag {};

	Term(int type): m_type(type) {}
	template<class T> Term(int type, T t): m_type(type), m_value(t) {}
	Term(int type, const Any& value): m_type(type), m_value(value) {}

	static Term Character(wchar32 c);
	static Term Repetition(int lower, int upper);
	static Term Dot();
	static Term BeginMark();
	static Term EndMark();

	int Type() const  { return m_type; }
	const Any& Value() const { return m_value; }
private:
	int m_type;
	Any m_value;
};

class Feature;

/**
* A class performing regexp pattern parsing.
*/
class Lexer {
public:
	// One-size-fits-all constructor set.
	Lexer()
		: m_encoding(&Encodings::Latin1())
	{ InstallDefaultFeatures(); }

	explicit Lexer(const char* str)
		: m_encoding(&Encodings::Latin1())
	{
		InstallDefaultFeatures();
		Assign(str, str + strlen(str));
	}
	template<class T> explicit Lexer(const T& t)
		: m_encoding(&Encodings::Latin1())
	{
		InstallDefaultFeatures();
		Assign(t.begin(), t.end());
	}

	template<class Iter> Lexer(Iter begin, Iter end)
		: m_encoding(&Encodings::Latin1())
	{
		InstallDefaultFeatures();
		Assign(begin, end);
	}
	~Lexer();

	template<class Iter> void Assign(Iter begin, Iter end)
	{
		m_input.clear();
		std::copy(begin, end, std::back_inserter(m_input));
	}

	/// The main lexer function. Extracts and returns the next term in input sequence.
	Term Lex();
	/// Installs an additional lexer feature.
	/// We declare both lvalue and rvalue reference types to fix some linker errors.
	Lexer& AddFeature(THolder<Feature>& a);
	Lexer& AddFeature(THolder<Feature>&& a);

	const Pire::Encoding& Encoding() const { return *m_encoding; }
	Lexer& SetEncoding(const Pire::Encoding& encoding) { m_encoding = &encoding; return *this; }
	void SetError(const char* msg) { errmsg = msg; }
	void SetError(ystring msg) { errmsg = msg; }
	ystring& GetError() { return errmsg; }

	Any& Retval() { return m_retval; }

	Fsm Parse();

	void Parenthesized(Fsm& fsm);

private:
	Term DoLex();

	wchar32 GetChar();
	wchar32 PeekChar();
	void UngetChar(wchar32 c);

	void Error(const char* msg) { throw Pire::Error(msg); }

	void InstallDefaultFeatures();

	TDeque<wchar32> m_input;
	const Pire::Encoding* m_encoding;
	TVector<THolder<Feature>> m_features;
	Any m_retval;
	ystring errmsg;

	friend class Feature;

	Lexer(const Lexer&);
	Lexer& operator = (const Lexer&);
};

/**
* A basic class for Pire customization.
* Features can be installed in the lexer and alter its behaviour.
*/
class Feature {
public:
	/// Precedence of features. The less the priority, the earlier
	/// will Lex() be called, and the later will Alter() and Parenthesized() be called.
	virtual int Priority() const { return 50; }

	/// Lexer will call this function to check whether the feature
	/// wants to handle the next part of the input sequence in its
	/// specific way. If it does not, features Lex() will not be called.
	virtual bool Accepts(wchar32 /*c*/) const { return false; }
	/// Should eat up some part of the input sequence, handle it
	/// somehow and produce a terminal.
	virtual Term Lex() { return Term(0); }

	/// This function recieves a shiny new terminal, and the feature
	/// has a chance to hack it somehow if it wants.
	virtual void Alter(Term&) {}
	/// This function recieves a parenthesized part of a pattern, and the feature
	/// has a chance to hack it somehow if it wants (its the way to implement
	/// those perl-style (?@#$%:..) clauses).
	virtual void Parenthesized(Fsm&) {}

	using Ptr = THolder<Feature>;

	virtual ~Feature() = default;

protected:

	// These functions are exposed versions of the corresponding lexer functions.
	const Pire::Encoding& Encoding() const { return m_lexer->Encoding(); }
	wchar32 GetChar() { return m_lexer->GetChar(); }
	wchar32 PeekChar() { return m_lexer->PeekChar(); }
	void UngetChar(wchar32 c) { m_lexer->UngetChar(c); }
	wchar32 CorrectChar(wchar32 c, const char* controls);
	void Error(const char* msg) { m_lexer->Error(msg); }

private:
	friend class Lexer;
	Lexer* m_lexer;
};

namespace Features {
	/// Disables case sensitivity
	Feature::Ptr CaseInsensitive();

	/**
	* Adds two more operations:
	*  (pattern1)&(pattern2) -- matches those strings which match both /pattern1/ and /pattern2/;
	*  ~(pattern)            -- matches those strings which do not match /pattern/.
	*/
	Feature::Ptr AndNotSupport();
}

}

#endif
