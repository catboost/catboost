#pragma once

#include <util/generic/deque.h>
#include <util/generic/hash.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/stream/buffer.h>
#include <util/memory/blob.h>
#include <utility>

#include <library/cpp/on_disk/chunks/writer.h>
#include <library/cpp/on_disk/chunks/chunked_helpers.h>

#include "common.h"

template <class I>
struct meta_iterator_pair {
    typedef typename std::iterator_traits<I>::value_type::first_type first_type;
    typedef typename std::iterator_traits<I>::value_type::second_type second_type;
};

/*
 * Builder implementation
 */

/*
 * Builder declaration
 */

template <class O>
class TDefaultContainerBuilder {
    TGeneralVectorWriter<O> List_;

public:
    void AddOut(const O& o) {
        List_.PushBack(o);
    }

    bool IsEmpty() const {
        return List_.Size() == 0;
    }

    void SaveContent(IOutputStream* stream) const {
        List_.Save(*stream);
    }
};

template <class O>
class TSingleContainerBuilder {
    bool Empty;
    O Out_;

public:
    TSingleContainerBuilder()
        : Empty(true)
    {
    }

    void AddOut(const O& o) {
        Empty = false;
        Out_ = o;
    }

    bool IsEmpty() const {
        return Empty;
    }

    void SaveContent(IOutputStream* stream) const {
        if (IsEmpty()) {
            WriteBin<ui32>(stream, 0);
        } else {
            TBuffer buf;
            {
                TBufferOutput tempStream(buf);
                TSaveLoadVectorNonPodElement<O>::Save(&tempStream, Out_);
            }
            WriteBin<ui32>(stream, buf.Size());
            stream->Write(buf.Data(), buf.Size());
        }
    }
};

template <class TStringType, class O, class C>
class TAhoCorasickBuilder;

template <class TStringType, class O, class C>
class TAhoVertex : TNonCopyable {
    typedef TAhoVertex<TStringType, O, C> TMyself;
    typedef TAhoCorasickBuilder<TStringType, O, C> TParent;
    typedef typename TStringType::value_type TCharType;

    friend class TAhoCorasickBuilder<TStringType, O, C>;

private:
    typedef THashMap<TCharType, TMyself*> TGotoMap;

    TGotoMap GotoMap_;
    C Output_;
    TMyself* FailVertex_;
    TMyself* SuffixVertex_;

protected:
    const C& Output() const {
        return Output_;
    }

    void AddVertex(TMyself* v, TCharType c) {
        GotoMap_.insert(std::make_pair(c, v));
    }

    TMyself* Fail() const {
        return FailVertex_;
    }

    TMyself* Suffix() const {
        return SuffixVertex_;
    }

    TMyself* GotoFunction(const TCharType c) {
        typename TGotoMap::iterator it;
        it = GotoMap_.find(c);
        return it != GotoMap_.end() ? it->second : nullptr;
    }

    TMyself const* GotoFunction(const TCharType c) const {
        typename TGotoMap::const_iterator it = GotoMap_.find(c);
        return it != GotoMap_.end() ? it->second : NULL;
    }

    TMyself* AddString(TParent* ahoCorasick, const TStringType& s, const ui32 position, const O& o) {
        if (position >= s.size()) {
            Output_.AddOut(o);
            return nullptr;
        } else {
            TCharType c = s[position];
            TMyself* v = GotoFunction(c);
            if (!v) {
                v = ahoCorasick->CreateAhoVertex();
                AddVertex(v, c);
            }
            return v;
        }
    }

    void SetFail(TMyself* v) {
        FailVertex_ = v;
    }

    void SetSuffix(TMyself* v) {
        SuffixVertex_ = v;
    }

    const TGotoMap& GotoMap() const {
        return GotoMap_;
    }

public:
    TAhoVertex()
        : FailVertex_(nullptr)
        , SuffixVertex_(nullptr)
    {
    }
};

template <class TStringType, class O, class C = TDefaultContainerBuilder<O>>
class TAhoCorasickBuilder : TNonCopyable {
public:
    typedef TAhoVertex<TStringType, O, C> TAhoVertexType;
    typedef typename TStringType::value_type TCharType;

    friend class TAhoVertex<TStringType, O, C>;
    friend class TTestMappedAhoCorasick;

private:
    TDeque<TAhoVertexType*> AhoVertexes;

private:
    TAhoVertexType* GetRoot() {
        return AhoVertexes.front();
    }

    TAhoVertexType const* GetRoot() const {
        return AhoVertexes.front();
    }

    TAhoVertexType* CreateAhoVertex() {
        AhoVertexes.push_back(new TAhoVertexType());
        return AhoVertexes.back();
    }

    void ConstructFail();

public:
    TAhoCorasickBuilder()
        : AhoVertexes(1, new TAhoVertexType())
    {
    }

    ~TAhoCorasickBuilder() {
        for (size_t i = 0; i < AhoVertexes.size(); ++i) {
            delete AhoVertexes[i];
        }
    }

    void AddString(const TStringType& s, const O& value) {
        TAhoVertexType* c = GetRoot();
        for (ui32 i = 0; i <= s.size(); ++i) {
            c = c->AddString(this, s, i, value);
        }
    }

    const TBlob Save();
    const TBlob AtomicSave();
    void SaveToStream(IOutputStream* stream);
};

using TSimpleAhoCorasickBuilder = TAhoCorasickBuilder<TString, ui32, TSingleContainerBuilder<ui32>>;
using TDefaultAhoCorasickBuilder = TAhoCorasickBuilder<TString, ui32>;

template <class AhoCorasick, class Iterator>
const TBlob BuildAho(AhoCorasick& ahoCorasick, Iterator begin, Iterator end) {
    for (Iterator it = begin; it != end; ++it)
        ahoCorasick.AddString(*it, it->size());
    return ahoCorasick.Save();
}

template <class TStringType, class Iterator>
const TBlob BuildAhoIndex(TAhoCorasickBuilder<TStringType, ui32>& ahoCorasick, Iterator begin, Iterator end) {
    ui32 index = 0;
    for (Iterator it = begin; it != end; ++it, ++index)
        ahoCorasick.AddString(*it, index);
    return ahoCorasick.Save();
}

template <class TStringType, class Iterator>
const TBlob BuildAhoObject(TAhoCorasickBuilder<TStringType, typename meta_iterator_pair<Iterator>::second_type>& ahoCorasick, Iterator begin, Iterator end) {
    for (Iterator it = begin; it != end; ++it)
        ahoCorasick.AddString(it->first, it->second);
    return ahoCorasick.Save();
}

template <class TStringType, class O, class C>
void TAhoCorasickBuilder<TStringType, O, C>::ConstructFail() {
    TAhoVertexType* root = GetRoot();
    root->SetFail(root);
    TDeque<TAhoVertexType*> q;
    typename TAhoVertexType::TGotoMap::const_iterator it;
    for (it = root->GotoMap().begin(); it != root->GotoMap().end(); ++it) {
        TAhoVertexType* v = it->second;
        v->SetFail(root);
        q.push_back(v);
    }
    while (!q.empty()) {
        TAhoVertexType* c = q.front();
        q.pop_front();
        for (it = c->GotoMap().begin(); it != c->GotoMap().end(); ++it) {
            TAhoVertexType* v = it->second;
            TCharType a = it->first;
            q.push_back(v);
            TAhoVertexType* h = c->Fail();
            bool outer = false;
            while (!h->GotoFunction(a)) {
                if (h->Fail() == h) {
                    v->SetFail(h);
                    outer = true;
                    break;
                }
                h = h->Fail();
            }
            if (outer)
                continue;
            TAhoVertexType* fail = h->GotoFunction(a);
            v->SetFail(fail);
            if (!fail->Output().IsEmpty())
                v->SetSuffix(fail);
            else
                v->SetSuffix(fail->Suffix());
        }
    }
}

template <class TStringType, class O, class C>
void TAhoCorasickBuilder<TStringType, O, C>::SaveToStream(IOutputStream* out) {
    ConstructFail(); /// the reason of non-const declaration

    Y_ASSERT(AhoVertexes.size() < Max<ui32>());
    const ui32 vertexAmount = (ui32)AhoVertexes.size();

    TChunkedDataWriter writer(*out);
    {
        TSingleValueWriter<ui32> versionWriter(TAhoCorasickCommon::GetVersion());
        WriteBlock(writer, versionWriter);
    }
    writer.NewBlock();

    TVector<TAhoVertexType const*> q(1, GetRoot());
    THashMap<TAhoVertexType const*, ui32> vertex2id(vertexAmount + 1);
    TVector<ui32> id2offset(vertexAmount, 0);

    TAhoVertexType* vt = nullptr;
    vertex2id[vt] = (ui32)-1;
    q.reserve(vertexAmount);

    for (ui32 curId = 0; curId < vertexAmount; ++curId) {
        TAhoVertexType const* c = q[curId];
        vertex2id[c] = curId;
        id2offset[curId] = (ui32)writer.GetCurrentBlockOffset();

        WriteBin<ui32>(&writer, vertex2id[c->Fail()]);
        WriteBin<ui32>(&writer, vertex2id[c->Suffix()]);

        typedef TVector<std::pair<TCharType, TAhoVertexType const*>> TChildren;
        TChildren children(c->GotoMap().begin(), c->GotoMap().end());
        WriteBin<ui32>(&writer, static_cast<ui32>(children.size()));

        if (!children.empty()) {
            TPlainHashWriter<TCharType, ui32> hashWriter;
            const ui32 id = static_cast<ui32>(q.size());
            for (size_t i = 0; i < children.size(); ++i) {
                hashWriter.Add(children[i].first, ui32(id + i));
                q.push_back(children[i].second);
            }

            hashWriter.Save(writer);
        }

        c->Output().SaveContent(&writer);
    }

    {
        Y_ASSERT(id2offset.size() < Max<ui32>());
        TSingleValueWriter<ui32> lenWriter((ui32)id2offset.size());
        WriteBlock(writer, lenWriter);
    }
    writer.NewBlock();
    writer.Write((const char*)id2offset.data(), id2offset.size() * sizeof(ui32));
    writer.WriteFooter();
    Y_ASSERT(TAhoCorasickCommon::GetBlockCount() == writer.GetBlockCount());
}

template <class TStringType, class O, class C>
const TBlob TAhoCorasickBuilder<TStringType, O, C>::Save() {
    TBufferStream buffer;
    SaveToStream(&buffer);
    return TBlob::FromStream(buffer);
}

template <class TStringType, class O, class C>
const TBlob TAhoCorasickBuilder<TStringType, O, C>::AtomicSave() {
    TBufferStream buffer;
    SaveToStream(&buffer);
    return TBlob::FromStream(buffer);
}
