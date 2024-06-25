/*
 * classes.cpp -- implementation for Pire::CharClasses feature.
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


#include <contrib/libs/pire/pire/stub/stl.h>
#include <contrib/libs/pire/pire/stub/singleton.h>
#include <contrib/libs/pire/pire/stub/noncopyable.h>
#include <contrib/libs/pire/pire/stub/utf8.h>

#include "re_lexer.h"

namespace Pire {

namespace {

	class CharClassesTable: private NonCopyable {
	private:
		class CharClass {
		public:
			CharClass() {}
			explicit CharClass(wchar32 ch) { m_bounds.push_back(ymake_pair(ch, ch)); }
			CharClass(wchar32 lower, wchar32 upper) { m_bounds.push_back(ymake_pair(lower, upper)); }

			CharClass& operator |= (const CharClass& cc)
			{
				std::copy(cc.m_bounds.begin(), cc.m_bounds.end(), std::back_inserter(m_bounds));
				return *this;
			}

			CharClass  operator |  (const CharClass& cc) const
			{
				CharClass r(*this);
				r |= cc;
				return r;
			}

			TSet<wchar32> ToSet() const
			{
				TSet<wchar32> ret;
				for (auto&& bound : m_bounds)
					for (wchar32 c = bound.first; c <= bound.second; ++c)
						ret.insert(c);
				return ret;
			}

		private:
			TVector<ypair<wchar32, wchar32> > m_bounds;
		};

	public:
		bool Has(wchar32 wc) const
		{
			return (m_classes.find(to_lower(wc & ~ControlMask)) != m_classes.end());
		}

		TSet<wchar32> Get(wchar32 wc) const
		{
			auto it = m_classes.find(to_lower(wc & ~ControlMask));
			if (it == m_classes.end())
				throw Error("Unknown character class");
			return it->second.ToSet();
		}

		CharClassesTable()
		{
			m_classes['l'] = CharClass('A', 'Z') | CharClass('a', 'z');
			m_classes['c']
				= CharClass(0x0410, 0x044F) // Russian capital A to Russan capital YA, Russian small A to Russian small YA
				| CharClass(0x0401)         // Russian capital Yo
				| CharClass(0x0451)         // Russian small Yo
				;

			m_classes['w'] = m_classes['l'] | m_classes['c'];
			m_classes['d'] = CharClass('0', '9');
			m_classes['s']
				= CharClass(' ') | CharClass('\t') | CharClass('\r') | CharClass('\n')
				| CharClass(0x00A0)         // Non-breaking space
				;

			// A few special classes which do not have any negation
			m_classes['n'] = CharClass('\n');
			m_classes['r'] = CharClass('\r');
			m_classes['t'] = CharClass('\t');
		}

		TMap<wchar32, CharClass> m_classes;
	};

	class CharClassesImpl: public Feature {
	public:
		CharClassesImpl(): m_table(Singleton<CharClassesTable>()) {}
		int Priority() const { return 10; }

		void Alter(Term& t)
		{
			if (t.Value().IsA<Term::CharacterRange>()) {
				const Term::CharacterRange& range = t.Value().As<Term::CharacterRange>();
				typedef Term::CharacterRange::first_type CharSet;
				const CharSet& old = range.first;
				CharSet altered;
				bool pos = false;
				bool neg = false;
				for (auto&& i : old)
					if (i.size() == 1 && (i[0] & ControlMask) == Control && m_table->Has(i[0])) {
						if (is_upper(i[0] & ~ControlMask))
							neg = true;
						else
							pos = true;

						TSet<wchar32> klass = m_table->Get(i[0]);
						for (auto&& j : klass)
							altered.insert(Term::String(1, j));
					} else
						altered.insert(i);

				if (neg && (pos || range.second))
					Error("Positive and negative character ranges mixed");
				t = Term(t.Type(), Term::CharacterRange(altered, neg || range.second));
			}
		}

	private:
		CharClassesTable* m_table;
	};

}

namespace Features {
	Feature::Ptr CharClasses() { return Feature::Ptr(new CharClassesImpl); }
}

}

