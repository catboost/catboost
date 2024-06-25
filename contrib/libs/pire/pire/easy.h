/*
 * easy.h -- Pire Easy facilities.
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


/**
 * For those who never reads documentation, does not need any mysterious features
 * there is a fast and easy way to start using Pire.
 *
 * Just type:
 *
 *    Pire::Regexp sc("pattern of (my regexp)*", Pire::UTF8 | Pire::I);
 *    if (sc.Matches("pattern of my regexp"))
 *        std::cout << "Hooray!" << std::endl;
 *
 * Or, to go more crazy:
 *
 *    if ("pattern of my regexp" ==~ sc)
 *        std::cout << "What a perversion..." << std::endl;
 *
 * Scanner's constructor takes a pattern and a "bitwise ORed" combination of "flags".
 * Available "flags" are:
 *    I        - case insensitivity;
 *    ANDNOT   - support for additional operations (& and ~) inside the pattern;
 *    UTF8     - treat pattern input sequence as UTF-8 (surprise!)
 *    LATIN1   - guess what?
 *
 * (In fact, those are not "flags" and not "bitwise ORed". See code for details.)
 */

#ifndef PIRE_EASY_H_INCLUDED
#define PIRE_EASY_H_INCLUDED

#include <iterator>

#include <contrib/libs/pire/pire/stub/stl.h>

#include "pire.h"
#include "vbitset.h"

namespace Pire {
	
template<class Arg> class Option;

class Options {
public:
	Options(): m_encoding(&Pire::Encodings::Latin1()) {}
	~Options() { Clear(); }
	
	void Add(const Pire::Encoding& encoding) { m_encoding = &encoding; }
	void Add(Feature::Ptr&& feature) { m_features.push_back(std::move(feature)); }
	
	struct Proxy {
		Options* const o;
		/*implicit*/ Proxy(Options* opts): o(opts) {}
	};
	operator Proxy() { return Proxy(this); }
	
	Options(Options& o): m_encoding(o.m_encoding) { m_features.swap(o.m_features); }
	Options& operator = (Options& o) { m_encoding = o.m_encoding; m_features = std::move(o.m_features); o.Clear(); return *this; }
	
	Options(Proxy p): m_encoding(p.o->m_encoding) { m_features.swap(p.o->m_features); }
	Options& operator = (Proxy p) { m_encoding = p.o->m_encoding; m_features = std::move(p.o->m_features); p.o->Clear(); return *this; }
	
	void Apply(Lexer& lexer)
	{
		lexer.SetEncoding(*m_encoding);
		for (auto&& i : m_features) {
			lexer.AddFeature(i);
			i = 0;
		}
		m_features.clear();
	}
	
	template<class ArgT>
	/*implicit*/ Options(const Option<ArgT>& option);
	
	const Pire::Encoding& Encoding() const { return *m_encoding; }

private:
	const Pire::Encoding* m_encoding;
	TVector<Feature::Ptr> m_features;
	
	void Clear()
	{
		m_features.clear();
	}
};

template<class Arg>
class Option {
public:
	typedef Arg (*Ctor)();

	Option(Ctor ctor): m_ctor(ctor) {}

	friend Options operator | (Options::Proxy options, const Option<Arg>& self)
	{
		Options ret(options);
		ret.Add((*self.m_ctor)());
		return ret;
	}
	
	template<class Arg2>
	friend Options operator | (const Option<Arg2>& a, const Option<Arg>& b)
	{
		return Options() | a | b;
	}

private:
	Ctor m_ctor;
};


extern const Option<const Encoding&> UTF8;
extern const Option<const Encoding&> LATIN1;

extern const Option<Feature::Ptr> I;
extern const Option<Feature::Ptr> ANDNOT;


class Regexp {
public:
	template<class Pattern>
	explicit Regexp(Pattern pattern, Options options = Options())
	{
		Init(PatternBounds(pattern), options);
	}
	
	template<class Pattern, class Arg>
	Regexp(Pattern pattern, Option<Arg> option)
	{
		Init(PatternBounds(pattern), Options() | option);
	}
	
	explicit Regexp(Scanner sc): m_scanner(sc) {}
	explicit Regexp(SlowScanner ssc): m_slow(ssc) {}
	
	bool Matches(TStringBuf buf) const
	{
		if (!m_scanner.Empty())
			return Runner(m_scanner).Begin().Run(buf).End();
		else
			return Runner(m_slow).Begin().Run(buf).End();
	}

	bool Matches(const char* begin, const char* end) const
	{
		return Matches(TStringBuf(begin, end));
	}
	
	/// A helper class allowing '==~' operator for regexps
	class MatchProxy {
	public:
		MatchProxy(const Regexp& re): m_re(&re) {}
		friend bool operator == (const char* str, const MatchProxy& re)    { return re.m_re->Matches(str); }
		friend bool operator == (const ystring& str, const MatchProxy& re) { return re.m_re->Matches(str); }

	private:
		const Regexp* m_re;
	};
	MatchProxy operator ~() const { return MatchProxy(*this); }
		
private:
	Scanner m_scanner;
	SlowScanner m_slow;
	
	ypair<const char*, const char*> PatternBounds(const ystring& pattern)
	{
		static const char c = 0;
		return pattern.empty() ? ymake_pair(&c, &c) : ymake_pair(pattern.c_str(), pattern.c_str() + pattern.size());
	}
	
	ypair<const char*, const char*> PatternBounds(const char* pattern)
	{
		return ymake_pair(pattern, pattern + strlen(pattern));
	}
	
	void Init(ypair<const char*, const char*> rawPattern, Options options)
	{
		TVector<wchar32> pattern;
		options.Encoding().FromLocal(rawPattern.first, rawPattern.second, std::back_inserter(pattern));
		
		Lexer lexer(pattern);
		options.Apply(lexer);
		Fsm fsm = lexer.Parse();
		
		if (!BeginsWithCircumflex(fsm))
			fsm.PrependAnything();
		fsm.AppendAnything();
		
		if (fsm.Determine())
			m_scanner = fsm.Compile<Scanner>();
		else
			m_slow = fsm.Compile<SlowScanner>();
	}
	
	static bool BeginsWithCircumflex(const Fsm& fsm)
	{
		typedef Fsm::StatesSet Set;
		TDeque<size_t> queue;
		BitSet handled(fsm.Size());
		
		queue.push_back(fsm.Initial());
		handled.Set(fsm.Initial());
		
		while (!queue.empty()) {
			Set s = fsm.Destinations(queue.front(), SpecialChar::Epsilon);
			for (auto&& i : s) {
				if (!handled.Test(i)) {
					handled.Set(i);
					queue.push_back(i);
				}
			}
			
			TSet<Char> lets = fsm.OutgoingLetters(queue.front());
			lets.erase(SpecialChar::Epsilon);
			lets.erase(SpecialChar::BeginMark);
			if (!lets.empty())
				return false;
			
			queue.pop_front();
		}
		
		return true;
	}	
};

};

#endif
