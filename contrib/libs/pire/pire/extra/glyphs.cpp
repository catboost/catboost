/*
 * glyphs.cpp -- implementation for the GlueSimilarGlyphs feature.
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


#include <algorithm>
#include <map>
#include <list>
#include <set>
#include <vector>
#include <utility>

#include <contrib/libs/pire/pire/stub/singleton.h>
#include <contrib/libs/pire/pire/stub/noncopyable.h>
#include <contrib/libs/pire/pire/stub/utf8.h>
#include <contrib/libs/pire/pire/stub/stl.h>
#include <contrib/libs/pire/pire/re_lexer.h>

namespace Pire {

namespace {

	/*
	* A class providing a function which returns a character
	* whose glyph resembles that of given char, if any;
	* otherwise returns given char itself.
	*/
	class GlyphTable {
	private:
		TList< TVector<wchar32> > m_classes;
		TMap<wchar32, TVector<wchar32>*> m_map;

		struct GlyphClass {
			TVector<wchar32>* m_class;
			TMap<wchar32, TVector<wchar32>*> *m_map;

			GlyphClass& operator << (wchar32 c)
			{
				m_class->push_back(c);
				m_map->insert(ymake_pair(c, m_class));
				return *this;
			}
		};

		GlyphClass Class()
		{
			GlyphClass cl;
			m_classes.push_back(TVector<wchar32>());
			cl.m_class = &m_classes.back();
			cl.m_map = &m_map;
			return cl;
		}

	public:

		const TVector<wchar32>& Klass(wchar32 x) const
		{
			TMap<wchar32, TVector<wchar32>*>::const_iterator i = m_map.find(x);
			if (i != m_map.end())
				return *i->second;
			else
				return DefaultValue< TVector<wchar32> >();
		}

		GlyphTable()
		{
			Class() << 'A' << 0x0410;
			Class() << 'B' << 0x0412;
			Class() << 'C' << 0x0421;
			Class() << 'E' << 0x0415 << 0x0401;
			Class() << 'H' << 0x041D;
			Class() << 'K' << 0x041A;
			Class() << 'M' << 0x041C;
			Class() << 'O' << 0x041E;
			Class() << 'P' << 0x0420;
			Class() << 'T' << 0x0422;
			Class() << 'X' << 0x0425;

			Class() << 'a' << 0x0430;
			Class() << 'c' << 0x0441;
			Class() << 'e' << 0x0435 << 0x0451;
			Class() << 'm' << 0x0442;
			Class() << 'o' << 0x043E;
			Class() << 'p' << 0x0440;
			Class() << 'u' << 0x0438;
			Class() << 'x' << 0x0445;
			Class() << 'y' << 0x0443;
		}
	};

	class GlueSimilarGlyphsImpl: public Feature {
	public:
		GlueSimilarGlyphsImpl(): m_table(Singleton<GlyphTable>()) {}
		int Priority() const { return 9; }

		void Alter(Term& t)
		{
			if (t.Value().IsA<Term::CharacterRange>()) {
				const Term::CharacterRange& range = t.Value().As<Term::CharacterRange>();
				typedef Term::CharacterRange::first_type CharSet;
				const CharSet& old = range.first;
				CharSet altered;
				for (auto&& i : old) {
					const TVector<wchar32>* klass = 0;
					if (i.size() == 1 && !(klass = &m_table->Klass(i[0]))->empty())
						for (auto&& j : *klass)
							altered.insert(Term::String(1, j));
					else
						altered.insert(i);
				}

				t = Term(t.Type(), Term::CharacterRange(altered, range.second));
			}
		}

	private:
		GlyphTable* m_table;
	};
}

namespace Features {
	Feature::Ptr GlueSimilarGlyphs() { return Feature::Ptr(new GlueSimilarGlyphsImpl); }
}

}

