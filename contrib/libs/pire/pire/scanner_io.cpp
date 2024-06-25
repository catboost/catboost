/*
 * scanner_io.cpp -- scanner serialization and deserialization
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
#include <contrib/libs/pire/pire/stub/saveload.h>
#include <contrib/libs/pire/pire/scanners/common.h>
#include <contrib/libs/pire/pire/scanners/slow.h>
#include <contrib/libs/pire/pire/scanners/simple.h>
#include <contrib/libs/pire/pire/scanners/loaded.h>

#include "align.h"

namespace Pire {
	
void SimpleScanner::Save(yostream* s) const
{
	SavePodType(s, Header(ScannerIOTypes::SimpleScanner, sizeof(m)));
	Impl::AlignSave(s, sizeof(Header));
	Locals mc = m;
	mc.initial -= reinterpret_cast<size_t>(m_transitions);
	SavePodType(s, mc);
	Impl::AlignSave(s, sizeof(mc));
	SavePodType(s, Empty());
	Impl::AlignSave(s, sizeof(Empty()));
	if (!Empty()) {
		Y_ASSERT(m_buffer);
		Impl::AlignedSaveArray(s, m_buffer.Get(), BufSize());
	}
}

void SimpleScanner::Load(yistream* s)
{
	SimpleScanner sc;
	Impl::ValidateHeader(s, ScannerIOTypes::SimpleScanner, sizeof(sc.m));
	LoadPodType(s, sc.m);
	Impl::AlignLoad(s, sizeof(sc.m));
	bool empty;
	LoadPodType(s, empty);
	Impl::AlignLoad(s, sizeof(empty));
	if (empty) {
		sc.Alias(Null());
	} else {
		sc.m_buffer = BufferType(new char[sc.BufSize()]);
		Impl::AlignedLoadArray(s, sc.m_buffer.Get(), sc.BufSize());
		sc.Markup(sc.m_buffer.Get());
		sc.m.initial += reinterpret_cast<size_t>(sc.m_transitions);
	}
	Swap(sc);
}

void SlowScanner::Save(yostream* s) const
{
	SavePodType(s, Header(ScannerIOTypes::SlowScanner, sizeof(m)));
	Impl::AlignSave(s, sizeof(Header));
	SavePodType(s, m);
	Impl::AlignSave(s, sizeof(m));
	SavePodType(s, Empty());
	Impl::AlignSave(s, sizeof(Empty()));
	if (!Empty()) {
		Y_ASSERT(!m_vec.empty());
		Impl::AlignedSaveArray(s, m_letters, MaxChar);
		Impl::AlignedSaveArray(s, m_finals, m.statesCount);

		size_t c = 0;
		SavePodType<size_t>(s, 0);
		for (auto&& i : m_vec) {
			size_t n = c + i.size();
			SavePodType(s, n);
			c = n;
		}
		Impl::AlignSave(s, (m_vec.size() + 1) * sizeof(size_t));

		size_t size = 0;
		for (auto&& i : m_vec)
			if (!i.empty()) {
				SavePodArray(s, &(i)[0], i.size());
				size += sizeof(unsigned) * i.size();
			}
		Impl::AlignSave(s, size);
		if (need_actions) {
			size_t pos = 0;
			for (TVector< TVector< Action > >::const_iterator i = m_actionsvec.begin(), ie = m_actionsvec.end(); i != ie; ++i)
				if (!i->empty()) {
					SavePodArray(s, &(*i)[0], i->size());
					pos += sizeof(Action) * i->size();
				}
			Impl::AlignSave(s, pos);
		}
	}
}

void SlowScanner::Load(yistream* s)
{
	SlowScanner sc;
	Impl::ValidateHeader(s, ScannerIOTypes::SlowScanner, sizeof(sc.m));
	LoadPodType(s, sc.m);
	Impl::AlignLoad(s, sizeof(sc.m));
	bool empty;
	LoadPodType(s, empty);
	Impl::AlignLoad(s, sizeof(empty));
	sc.need_actions = need_actions;
	if (empty) {
		sc.Alias(Null());
	} else {
		sc.m_vec.resize(sc.m.lettersCount * sc.m.statesCount);
		if (sc.need_actions)
			sc.m_actionsvec.resize(sc.m.lettersCount * sc.m.statesCount);
		sc.m_vecptr = &sc.m_vec;

		sc.alloc(sc.m_letters, MaxChar);
		Impl::AlignedLoadArray(s, sc.m_letters, MaxChar);

		sc.alloc(sc.m_finals, sc.m.statesCount);
		Impl::AlignedLoadArray(s, sc.m_finals, sc.m.statesCount);

		size_t c;
		LoadPodType(s, c);
		auto act = sc.m_actionsvec.begin();
		for (auto&& i : sc.m_vec) {
			size_t n;
			LoadPodType(s, n);
			i.resize(n - c);
			if (sc.need_actions) {
				act->resize(n - c);
				++act;
			}
			c = n;
		}
		Impl::AlignLoad(s, (m_vec.size() + 1) * sizeof(size_t));

		size_t size = 0;
		for (auto&& i : sc.m_vec)
			if (!i.empty()) {
				LoadPodArray(s, &(i)[0], i.size());
				size += sizeof(unsigned) * i.size();
			}
		Impl::AlignLoad(s, size);
		size_t actSize = 0;
		if (sc.need_actions) {
			for (auto&& i : sc.m_actionsvec) {
				if (!i.empty()) {
					LoadPodArray(s, &(i)[0], i.size());
					actSize += sizeof(Action) * i.size();
				}
			}
			Impl::AlignLoad(s, actSize);
		}
	}
	Swap(sc);
}

void LoadedScanner::Save(yostream* s) const {
	Save(s, ScannerIOTypes::LoadedScanner);
}

void LoadedScanner::Save(yostream* s, ui32 type) const
{
	Y_ASSERT(type == ScannerIOTypes::LoadedScanner || type == ScannerIOTypes::NoGlueLimitCountingScanner);
	SavePodType(s, Header(type, sizeof(m)));
	Impl::AlignSave(s, sizeof(Header));
	Locals mc = m;
	mc.initial -= reinterpret_cast<size_t>(m_jumps);
	SavePodType(s, mc);
	Impl::AlignSave(s, sizeof(mc));

	Impl::AlignedSaveArray(s, m_letters, MaxChar);	
	Impl::AlignedSaveArray(s, m_jumps, m.statesCount * m.lettersCount);	
	Impl::AlignedSaveArray(s, m_tags, m.statesCount);
}

void LoadedScanner::Load(yistream* s) {
	Load(s, nullptr);
}

void LoadedScanner::Load(yistream* s, ui32* type)
{
	LoadedScanner sc;
	Header header = Impl::ValidateHeader(s, ScannerIOTypes::LoadedScanner, sizeof(sc.m));
	if (type) {
		*type = header.Type;
	}
	LoadPodType(s, sc.m);
	Impl::AlignLoad(s, sizeof(sc.m));
	sc.m_buffer = BufferType(new char[sc.BufSize()]);
	sc.Markup(sc.m_buffer.Get());
	Impl::AlignedLoadArray(s, sc.m_letters, MaxChar);
	Impl::AlignedLoadArray(s, sc.m_jumps, sc.m.statesCount * sc.m.lettersCount);
	if (header.Version == Header::RE_VERSION_WITH_MACTIONS) {
		TVector<Action> actions(sc.m.statesCount * sc.m.lettersCount);
		Impl::AlignedLoadArray(s, actions.data(), actions.size());
	}
	Impl::AlignedLoadArray(s, sc.m_tags, sc.m.statesCount);
	sc.m.initial += reinterpret_cast<size_t>(sc.m_jumps);
	Swap(sc);
}

}
