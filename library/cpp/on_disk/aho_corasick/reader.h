#pragma once

#include <util/generic/deque.h>
#include <util/generic/strbuf.h>
#include <util/generic/yexception.h>
#include <util/memory/blob.h>
#include <util/stream/buffer.h>
#include <util/stream/mem.h>
#include <util/system/unaligned_mem.h>
#include <utility>

#include <library/cpp/on_disk/chunks/chunked_helpers.h>

#include "common.h"

template <class O>
class TAhoSearchResult: public TDeque<std::pair<ui32, O>> {
};

/*
 * Mapped-declaraion
 */

template <class O>
class TMappedDefaultOutputContainer {
private:
    TGeneralVector<O> List_;

public:
    TMappedDefaultOutputContainer(const char* data)
        : List_(TBlob::NoCopy(data, (size_t)-1))
    {
    }

    bool IsEmpty() const {
        return List_.GetSize() == 0;
    }

    void FillAnswer(TAhoSearchResult<O>& answer, ui32 pos) const {
        for (ui32 i = 0; i < List_.GetSize(); ++i) {
            answer.push_back(std::make_pair(pos, O()));
            List_.Get(i, answer.back().second);
        }
    }

    size_t CheckData() const {
        return List_.RealSize();
    }
};

template <class O>
class TMappedSingleOutputContainer {
    const ui32* Data;

    ui32 Size() const {
        return ReadUnaligned<ui32>(Data);
    }

public:
    TMappedSingleOutputContainer(const char* data)
        : Data((const ui32*)data)
    {
    }

    bool IsEmpty() const {
        return Size() == 0;
    }

    void FillAnswer(TAhoSearchResult<O>& answer, ui32 pos) const {
        if (!IsEmpty()) {
            answer.push_back(std::make_pair(pos, O()));
            TMemoryInput input(Data + 1, Size());
            TSaveLoadVectorNonPodElement<O>::Load(&input, answer.back().second, Size());
        }
    }

    size_t CheckData() const {
        return sizeof(ui32) + ReadUnaligned<ui32>(Data);
    }
};

template <class TStringType, class O, class C>
class TMappedAhoCorasick;

template <typename TKey, typename TValue>
class TEmptyMapData : TNonCopyable {
private:
    TBufferStream Stream;

public:
    const char* P;

    TEmptyMapData() {
        TPlainHashWriter<TKey, TValue> hash;
        hash.Save(Stream);
        P = Stream.Buffer().Begin();
    }
};

/*
 * каждая вершина имеет свой ui32-номер
 * блок данных для вершины:
 * ui32, ui32, ui32, ui32, степень*char, данные контейнера
 * fail, suff, степень, самый левый сын, лексикографический список меток исходящих рёбер.
 * если степень нулевая, то в блоке только 3 инта
 */
template <class TStringType, class O, class C>
class TMappedAhoVertex {
public:
    typedef typename TStringType::value_type TCharType;
    friend class TMappedAhoCorasick<TStringType, O, C>;

private:
    const char* Data;
    typedef TPlainHash<TCharType, ui32> TGotoMap;
    TGotoMap GotoMap;
    static const TEmptyMapData<TCharType, ui32> EmptyData;

    static const size_t GENERAL_SHIFT = 3 * sizeof(ui32);

private:
    const ui32* DataAsInt() const {
        return (const ui32*)Data;
    }

    ui32 Power() const {
        return ReadUnaligned<ui32>(DataAsInt() + 2);
    }

protected:
    const C Output() const {
        return C(Power() ? GotoMap.ByteEnd() : Data + GENERAL_SHIFT);
    }

    ui32 Fail() const {
        return ReadUnaligned<ui32>(DataAsInt());
    }

    ui32 Suffix() const {
        return ReadUnaligned<ui32>(DataAsInt() + 1);
    }

    bool GotoFunction(const TCharType c, ui32* result) const {
        if (0 == Power())
            return false;
        return GotoMap.Find(c, result);
    }

    bool operator==(const TMappedAhoVertex& rhs) const {
        return Data == rhs.Data;
    }

    size_t CheckData(ui32 totalVertices) const; /// throws yexception in case of bad data

public:
    TMappedAhoVertex(const char* data)
        : Data(data)
        , GotoMap(Power() ? (Data + GENERAL_SHIFT) : EmptyData.P)
    {
    }
};

/*
 * блок данных для бора:
 *   количество вершин N, ui32
 *   суммарный размер блоков для вершин, ui32
 *   блоки данных для каждой вершины
 *   отображение id->offset для блока вершины id, N*ui32
 */
template <class TStringType, class O, class C = TMappedDefaultOutputContainer<O>>
class TMappedAhoCorasick : TNonCopyable {
public:
    typedef TAhoSearchResult<O> TSearchResult;
    typedef TMappedAhoVertex<TStringType, O, C> TAhoVertexType;
    typedef typename TStringType::value_type TCharType;
    typedef TBasicStringBuf<TCharType> TSample;

private:
    const TBlob Blob;
    const char* const AhoVertexes;
    const ui32 VertexAmount;
    const ui32* const Id2Offset;
    const TAhoVertexType Root;

private:
    bool ValidVertex(ui32 id) const {
        return id < VertexAmount;
    }

    TAhoVertexType GetVertexAt(ui32 id) const {
        if (!ValidVertex(id))
            ythrow yexception() << "TMappedAhoCorasick fatal error: invalid id " << id;
        return TAhoVertexType(AhoVertexes + Id2Offset[id]);
    }

public:
    TMappedAhoCorasick(const TBlob& blob)
        : Blob(blob)
        , AhoVertexes(GetBlock(blob, 1).AsCharPtr())
        , VertexAmount(TSingleValue<ui32>(GetBlock(blob, 2)).Get())
        , Id2Offset((const ui32*)(GetBlock(Blob, 3).AsCharPtr()))
        , Root(GetVertexAt(0))
    {
        {
            const ui32 version = TSingleValue<ui32>(GetBlock(blob, 0)).Get();
            if (version != TAhoCorasickCommon::GetVersion())
                ythrow yexception() << "Unknown version " << version << " instead of " << TAhoCorasickCommon::GetVersion();
        }
        {
            TChunkedDataReader reader(blob);
            if (reader.GetBlocksCount() != TAhoCorasickCommon::GetBlockCount())
                ythrow yexception() << "wrong block count " << reader.GetBlocksCount();
        }
    }

    bool AhoContains(const TSample& str) const;
    TSearchResult AhoSearch(const TSample& str) const;
    void AhoSearch(const TSample& str, TSearchResult* result) const;
    size_t CheckData() const; /// throws yexception in case of bad data
};

using TSimpleMappedAhoCorasick = TMappedAhoCorasick<TString, ui32, TMappedSingleOutputContainer<ui32>>;
using TDefaultMappedAhoCorasick = TMappedAhoCorasick<TString, ui32>;

/*
 * Mapped-implementation
 */
template <class TStringType, class O, class C>
bool TMappedAhoCorasick<TStringType, O, C>::AhoContains(const TSample& str) const {
    TAhoVertexType current = Root;
    const size_t len = str.size();
    for (size_t i = 0; i < len; ++i) {
        bool outer = false;
        ui32 gotoVertex;
        while (!current.GotoFunction(str[i], &gotoVertex)) {
            if (current == Root) { /// nowhere to go
                outer = true;
                break;
            }
            current = GetVertexAt(current.Fail());
        }
        if (outer)
            continue;
        current = GetVertexAt(gotoVertex);

        TAhoVertexType v = current;
        while (true) {
            if (!v.Output().IsEmpty())
                return true;
            if ((ui32)-1 == v.Suffix())
                break;
            v = GetVertexAt(v.Suffix());
        }
    }
    return false;
}

template <class TStringType, class O, class C>
void TMappedAhoCorasick<TStringType, O, C>::AhoSearch(const TSample& str, typename TMappedAhoCorasick<TStringType, O, C>::TSearchResult* answer) const {
    answer->clear();
    TAhoVertexType current = Root;
    const size_t len = str.length();
    for (size_t i = 0; i < len; ++i) {
        bool outer = false;
        ui32 gotoVertex;
        while (!current.GotoFunction(str[i], &gotoVertex)) {
            if (current == Root) { /// nowhere to go
                outer = true;
                break;
            }
            current = GetVertexAt(current.Fail());
        }
        if (outer)
            continue;
        current = GetVertexAt(gotoVertex);

        TAhoVertexType v = current;
        while (true) {
            v.Output().FillAnswer(*answer, (ui32)i);
            if ((ui32)-1 == v.Suffix())
                break;
            v = GetVertexAt(v.Suffix());
        }
    }
}

template <class TStringType, class O, class C>
typename TMappedAhoCorasick<TStringType, O, C>::TSearchResult TMappedAhoCorasick<TStringType, O, C>::AhoSearch(const TSample& str) const {
    TAhoSearchResult<O> answer;
    AhoSearch(str, &answer);
    return answer;
}

/*
 * implementation of CheckData in Mapped-classes
 */

static inline void CheckRange(ui32 id, ui32 strictUpperBound) {
    if (id >= strictUpperBound) {
        throw yexception() << id << " of " << strictUpperBound << " - index is invalid";
    }
}

template <class TStringType, class O, class C>
const TEmptyMapData<typename TStringType::value_type, ui32> TMappedAhoVertex<TStringType, O, C>::EmptyData;

template <class TStringType, class O, class C>
size_t TMappedAhoVertex<TStringType, O, C>::CheckData(ui32 totalVertices) const {
    size_t bytesNeeded = GENERAL_SHIFT;
    CheckRange(Fail(), totalVertices);
    if (Suffix() != (ui32)(-1))
        CheckRange(Suffix(), totalVertices);
    if (Power()) {
        for (typename TGotoMap::TConstIterator toItem = GotoMap.Begin(); toItem != GotoMap.End(); ++toItem)
            CheckRange(toItem->Second(), totalVertices);
        bytesNeeded += GotoMap.ByteSize();
    }
    bytesNeeded += Output().CheckData();
    return bytesNeeded;
}

template <class TStringType, class O, class C>
size_t TMappedAhoCorasick<TStringType, O, C>::CheckData() const {
    try {
        size_t bytesNeeded = 0;
        for (ui32 id = 0; id < VertexAmount; ++id) {
            if (Id2Offset[id] != bytesNeeded) {
                ythrow yexception() << "wrong offset[" << id << "]: " << Id2Offset[id];
            }
            bytesNeeded += GetVertexAt(id).CheckData(VertexAmount);
        }
        bytesNeeded += VertexAmount * sizeof(ui32);
        const size_t realsize = GetBlock(Blob, 1).Size() + GetBlock(Blob, 3).Size();
        if (realsize != bytesNeeded) {
            ythrow yexception() << "extra information " << bytesNeeded << " " << realsize;
        }
        return bytesNeeded;
    } catch (const yexception& e) {
        ythrow yexception() << "Bad data: " << e.what();
    }
}
