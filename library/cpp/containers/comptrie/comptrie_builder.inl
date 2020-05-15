#pragma once

#include "comptrie_impl.h"
#include "comptrie_trie.h"
#include "make_fast_layout.h"
#include "array_with_size.h"

#include <library/cpp/containers/compact_vector/compact_vector.h>

#include <util/memory/alloc.h>
#include <util/memory/blob.h>
#include <util/memory/pool.h>
#include <util/memory/tempbuf.h>
#include <util/memory/smallobj.h>
#include <util/generic/algorithm.h>
#include <util/generic/buffer.h>
#include <util/generic/strbuf.h>

#include <util/system/align.h>
#include <util/stream/buffer.h>

#define CONSTEXPR_MAX2(a, b) (a) > (b) ? (a) : (b)
#define CONSTEXPR_MAX3(a, b, c) CONSTEXPR_MAX2(CONSTEXPR_MAX2(a, b), c)

// TCompactTrieBuilder::TCompactTrieBuilderImpl

template <class T, class D, class S>
class TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl {
protected:
    TMemoryPool Pool;
    size_t PayloadSize;
    THolder<TFixedSizeAllocator> NodeAllocator;
    class TNode;
    class TArc;
    TNode* Root;
    TCompactTrieBuilderFlags Flags;
    size_t EntryCount;
    size_t NodeCount;
    TPacker Packer;

    enum EPayload {
        DATA_ABSENT,
        DATA_INSIDE,
        DATA_MALLOCED,
        DATA_IN_MEMPOOL,
    };

protected:
    void ConvertSymbolArrayToChar(const TSymbol* key, size_t keylen, TTempBuf& buf, size_t ckeylen) const;
    void NodeLinkTo(TNode* thiz, const TBlob& label, TNode* node);
    TNode* NodeForwardAdd(TNode* thiz, const char* label, size_t len, size_t& passed, size_t* nodeCount);
    bool FindEntryImpl(const char* key, size_t keylen, TData* value) const;
    bool FindLongestPrefixImpl(const char* keyptr, size_t keylen, size_t* prefixLen, TData* value) const;

    size_t NodeMeasureSubtree(TNode* thiz) const;
    ui64 NodeSaveSubtree(TNode* thiz, IOutputStream& os) const;
    ui64 NodeSaveSubtreeAndDestroy(TNode* thiz, IOutputStream& osy);
    void NodeBufferSubtree(TNode* thiz);

    size_t NodeMeasureLeafValue(TNode* thiz) const;
    ui64 NodeSaveLeafValue(TNode* thiz, IOutputStream& os) const;

    virtual ui64 ArcMeasure(const TArc* thiz, size_t leftsize, size_t rightsize) const;

    virtual ui64 ArcSaveSelf(const TArc* thiz, IOutputStream& os) const;
    ui64 ArcSave(const TArc* thiz, IOutputStream& os) const;
    ui64 ArcSaveAndDestroy(const TArc* thiz, IOutputStream& os);

public:
    TCompactTrieBuilderImpl(TCompactTrieBuilderFlags flags, TPacker packer, IAllocator* alloc);
    virtual ~TCompactTrieBuilderImpl();

    void DestroyNode(TNode* node);
    void NodeReleasePayload(TNode* thiz);

    char* AddEntryForData(const TSymbol* key, size_t keylen, size_t dataLen, bool& isNewAddition);
    TNode* AddEntryForSomething(const TSymbol* key, size_t keylen, bool& isNewAddition);

    bool AddEntry(const TSymbol* key, size_t keylen, const TData& value);
    bool AddEntryPtr(const TSymbol* key, size_t keylen, const char* value);
    bool AddSubtreeInFile(const TSymbol* key, size_t keylen, const TString& fileName);
    bool AddSubtreeInBuffer(const TSymbol* key, size_t keylen, TArrayWithSizeHolder<char>&& buffer);
    bool FindEntry(const TSymbol* key, size_t keylen, TData* value) const;
    bool FindLongestPrefix(const TSymbol* key, size_t keylen, size_t* prefixlen, TData* value) const;

    size_t Save(IOutputStream& os) const;
    size_t SaveAndDestroy(IOutputStream& os);

    void Clear();

    // lies if some key was added at least twice
    size_t GetEntryCount() const;
    size_t GetNodeCount() const;

    size_t MeasureByteSize() const {
        return NodeMeasureSubtree(Root);
    }
};

template <class T, class D, class S>
class TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::TArc {
public:
    TBlob Label;
    TNode* Node;
    mutable size_t LeftOffset;
    mutable size_t RightOffset;

    TArc(const TBlob& lbl, TNode* nd);
};

template <class T, class D, class S>
class TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::TNode {
public:
    typedef typename TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl TBuilderImpl;
    typedef typename TBuilderImpl::TArc TArc;

    struct ISubtree {
        virtual ~ISubtree() = default;
        virtual bool IsLast() const = 0;
        virtual ui64 Measure(const TBuilderImpl* builder) const = 0;
        virtual ui64 Save(const TBuilderImpl* builder, IOutputStream& os) const = 0;
        virtual ui64 SaveAndDestroy(TBuilderImpl* builder, IOutputStream& os) = 0;
        virtual void Destroy(TBuilderImpl*) { }

        // Tries to find key in subtree.
        // Returns next node to find the key in (to avoid recursive calls)
        // If it has end result, writes it to @value and @result arguments and returns nullptr
        virtual const TNode* Find(TStringBuf& key, TData* value, bool& result, const TPacker& packer) const = 0;
        virtual const TNode* FindLongestPrefix(TStringBuf& key, TData* value, bool& result, const TPacker& packer) const = 0;
    };

    class TArcSet: public ISubtree, public TCompactVector<TArc> {
    public:
        typedef typename TCompactVector<TArc>::iterator iterator;
        typedef typename TCompactVector<TArc>::const_iterator const_iterator;

        TArcSet() {
            Y_ASSERT(reinterpret_cast<ISubtree*>(this) == static_cast<void*>(this)); // This assumption is used in TNode::Subtree()
        }

        iterator Find(char ch);
        const_iterator Find(char ch) const;
        void Add(const TBlob& s, TNode* node);

        bool IsLast() const override {
            return this->Empty();
        }

        const TNode* Find(TStringBuf& key, TData* value, bool& result, const TPacker& packer) const override;
        const TNode* FindLongestPrefix(TStringBuf& key, TData* value, bool& result, const TPacker& packer) const override {
            return Find(key, value, result, packer);
        }

        ui64 Measure(const TBuilderImpl* builder) const override {
            return MeasureRange(builder, 0, this->size());
        }

        ui64 MeasureRange(const TBuilderImpl* builder, size_t from, size_t to) const {
            if (from >= to)
                return 0;

            size_t median = (from + to) / 2;
            size_t leftsize = (size_t)MeasureRange(builder, from, median);
            size_t rightsize = (size_t)MeasureRange(builder, median + 1, to);

            return builder->ArcMeasure(&(*this)[median], leftsize, rightsize);
        }

        ui64 Save(const TBuilderImpl* builder, IOutputStream& os) const override {
            return SaveRange(builder, 0, this->size(), os);
        }

        ui64 SaveAndDestroy(TBuilderImpl* builder, IOutputStream& os) override {
            ui64 result = SaveRangeAndDestroy(builder, 0, this->size(), os);
            Destroy(builder);
            return result;
        }

        ui64 SaveRange(const TBuilderImpl* builder, size_t from, size_t to, IOutputStream& os) const {
            if (from >= to)
                return 0;

            size_t median = (from + to) / 2;

            ui64 written = builder->ArcSave(&(*this)[median], os);
            written += SaveRange(builder, from, median, os);
            written += SaveRange(builder, median + 1, to, os);
            return written;
        }

        ui64 SaveRangeAndDestroy(TBuilderImpl* builder, size_t from, size_t to, IOutputStream& os) {
            if (from >= to)
                return 0;

            size_t median = (from + to) / 2;

            ui64 written = builder->ArcSaveAndDestroy(&(*this)[median], os);
            written += SaveRangeAndDestroy(builder, from, median, os);
            written += SaveRangeAndDestroy(builder, median + 1, to, os);
            return written;
        }

        void Destroy(TBuilderImpl* builder) override {
            // Delete all nodes down the stream.
            for (iterator it = this->begin(); it != this->end(); ++it) {
                builder->DestroyNode(it->Node);
            }
            this->clear();
        }

        ~TArcSet() override {
            Y_ASSERT(this->empty());
        }

    };

    struct TBufferedSubtree: public ISubtree {
        TArrayWithSizeHolder<char> Buffer;

        TBufferedSubtree() {
            Y_ASSERT(reinterpret_cast<ISubtree*>(this) == static_cast<void*>(this)); // This assumption is used in TNode::Subtree()
        }

        bool IsLast() const override {
            return Buffer.Empty();
        }

        const TNode* Find(TStringBuf& key, TData* value, bool& result, const TPacker& packer) const override {
            if (Buffer.Empty()) {
                result = false;
                return nullptr;
            }

            TCompactTrie<char, D, S> trie(Buffer.Get(), Buffer.Size(), packer);
            result = trie.Find(key.data(), key.size(), value);

            return nullptr;
        }

        const TNode* FindLongestPrefix(TStringBuf& key, TData* value, bool& result, const TPacker& packer) const override {
            if (Buffer.Empty()) {
                result = false;
                return nullptr;
            }

            TCompactTrie<char, D, S> trie(Buffer.Get(), Buffer.Size(), packer);
            size_t prefixLen = 0;
            result = trie.FindLongestPrefix(key.data(), key.size(), &prefixLen, value);
            key = key.SubStr(prefixLen);

            return nullptr;
        }

        ui64 Measure(const TBuilderImpl*) const override {
            return Buffer.Size();
        }

        ui64 Save(const TBuilderImpl*, IOutputStream& os) const override {
            os.Write(Buffer.Get(), Buffer.Size());
            return Buffer.Size();
        }

        ui64 SaveAndDestroy(TBuilderImpl* builder, IOutputStream& os) override {
            ui64 result = Save(builder, os);
            TArrayWithSizeHolder<char>().Swap(Buffer);
            return result;
        }
    };

    struct TSubtreeInFile: public ISubtree {
        struct TData {
            TString FileName;
            ui64 Size;
        };
        THolder<TData> Data;

        TSubtreeInFile(const TString& fileName) {
            // stupid API
            TFile file(fileName, RdOnly);
            i64 size = file.GetLength();
            if (size < 0)
                ythrow yexception() << "unable to get file " << fileName.Quote() << " size for unknown reason";
            Data.Reset(new TData);
            Data->FileName = fileName;
            Data->Size = size;

            Y_ASSERT(reinterpret_cast<ISubtree*>(this) == static_cast<void*>(this)); // This assumption is used in TNode::Subtree()
        }

        bool IsLast() const override {
            return Data->Size == 0;
        }

        const TNode* Find(TStringBuf& key, typename TCompactTrieBuilder::TData* value, bool& result, const TPacker& packer) const override {
            if (!Data) {
                result = false;
                return nullptr;
            }

            TCompactTrie<char, D, S> trie(TBlob::FromFile(Data->FileName), packer);
            result = trie.Find(key.data(), key.size(), value);
            return nullptr;
        }

        const TNode* FindLongestPrefix(TStringBuf& key, typename TCompactTrieBuilder::TData* value, bool& result, const TPacker& packer) const override {
            if (!Data) {
                result = false;
                return nullptr;
            }

            TCompactTrie<char, D, S> trie(TBlob::FromFile(Data->FileName), packer);
            size_t prefixLen = 0;
            result = trie.FindLongestPrefix(key.data(), key.size(), &prefixLen, value);
            key = key.SubStr(prefixLen);

            return nullptr;
        }

        ui64 Measure(const TBuilderImpl*) const override {
            return Data->Size;
        }

        ui64 Save(const TBuilderImpl*, IOutputStream& os) const override {
            TUnbufferedFileInput is(Data->FileName);
            ui64 written = TransferData(&is, &os);
            if (written != Data->Size)
                ythrow yexception() << "file " << Data->FileName.Quote() << " size changed";
            return written;
        }

        ui64 SaveAndDestroy(TBuilderImpl* builder, IOutputStream& os) override {
            return Save(builder, os);
        }
    };

    union {
        char ArcsData[CONSTEXPR_MAX3(sizeof(TArcSet), sizeof(TBufferedSubtree), sizeof(TSubtreeInFile))];
        union {
            void* Data1;
            long long int Data2;
        } Aligner;
    };

    inline ISubtree* Subtree() {
        return reinterpret_cast<ISubtree*>(ArcsData);
    }

    inline const ISubtree* Subtree() const {
        return reinterpret_cast<const ISubtree*>(ArcsData);
    }

    EPayload PayloadType;

    inline const char* PayloadPtr() const {
        return ((const char*) this) + sizeof(TNode);
    }

    inline char* PayloadPtr() {
        return ((char*) this) + sizeof(TNode);
    }

    // *Payload()
    inline const char*& PayloadAsPtr() const {
        const char** payload = (const char**) PayloadPtr();
        return *payload;
    }

    inline char*& PayloadAsPtr() {
        char** payload = (char**) PayloadPtr();
        return *payload;
    }

    inline const char* GetPayload() const {
        switch (PayloadType) {
        case DATA_INSIDE:
            return PayloadPtr();
        case DATA_MALLOCED:
        case DATA_IN_MEMPOOL:
            return PayloadAsPtr();
        case DATA_ABSENT:
        default:
            abort();
        }
    }

    inline char* GetPayload() {
        const TNode* thiz = this;
        return const_cast<char*>(thiz->GetPayload()); // const_cast is to avoid copy-paste style
    }

    bool IsFinal() const {
        return PayloadType != DATA_ABSENT;
    }

    bool IsLast() const {
        return Subtree()->IsLast();
    }

    inline void* operator new(size_t, TFixedSizeAllocator& pool) {
        return pool.Allocate();
    }

    inline void operator delete(void* ptr, TFixedSizeAllocator& pool) noexcept {
        pool.Release(ptr);
    }

    TNode()
        : PayloadType(DATA_ABSENT)
    {
        new (Subtree()) TArcSet;
    }

    ~TNode() {
        Subtree()->~ISubtree();
        Y_ASSERT(PayloadType == DATA_ABSENT);
    }

};

// TCompactTrieBuilder

template <class T, class D, class S>
TCompactTrieBuilder<T, D, S>::TCompactTrieBuilder(TCompactTrieBuilderFlags flags, TPacker packer, IAllocator* alloc)
    : Impl(new TCompactTrieBuilderImpl(flags, packer, alloc))
{
}

template <class T, class D, class S>
bool TCompactTrieBuilder<T, D, S>::Add(const TSymbol* key, size_t keylen, const TData& value) {
    return Impl->AddEntry(key, keylen, value);
}

template <class T, class D, class S>
bool TCompactTrieBuilder<T, D, S>::AddPtr(const TSymbol* key, size_t keylen, const char* value) {
    return Impl->AddEntryPtr(key, keylen, value);
}

template <class T, class D, class S>
bool TCompactTrieBuilder<T, D, S>::AddSubtreeInFile(const TSymbol* key, size_t keylen, const TString& fileName) {
    return Impl->AddSubtreeInFile(key, keylen, fileName);
}

template <class T, class D, class S>
bool TCompactTrieBuilder<T, D, S>::AddSubtreeInBuffer(const TSymbol* key, size_t keylen, TArrayWithSizeHolder<char>&& buffer) {
    return Impl->AddSubtreeInBuffer(key, keylen, std::move(buffer));
}

template <class T, class D, class S>
bool TCompactTrieBuilder<T, D, S>::Find(const TSymbol* key, size_t keylen, TData* value) const {
    return Impl->FindEntry(key, keylen, value);
}

template <class T, class D, class S>
bool TCompactTrieBuilder<T, D, S>::FindLongestPrefix(
                const TSymbol* key, size_t keylen, size_t* prefixlen, TData* value) const {
    return Impl->FindLongestPrefix(key, keylen, prefixlen, value);
}

template <class T, class D, class S>
size_t TCompactTrieBuilder<T, D, S>::Save(IOutputStream& os) const {
    return Impl->Save(os);
}

template <class T, class D, class S>
size_t TCompactTrieBuilder<T, D, S>::SaveAndDestroy(IOutputStream& os) {
    return Impl->SaveAndDestroy(os);
}

template <class T, class D, class S>
void TCompactTrieBuilder<T, D, S>::Clear() {
    Impl->Clear();
}

template <class T, class D, class S>
size_t TCompactTrieBuilder<T, D, S>::GetEntryCount() const {
    return Impl->GetEntryCount();
}

template <class T, class D, class S>
size_t TCompactTrieBuilder<T, D, S>::GetNodeCount() const {
    return Impl->GetNodeCount();
}

// TCompactTrieBuilder::TCompactTrieBuilderImpl

template <class T, class D, class S>
TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::TCompactTrieBuilderImpl(TCompactTrieBuilderFlags flags, TPacker packer, IAllocator* alloc)
    : Pool(1000000, TMemoryPool::TLinearGrow::Instance(), alloc)
    , PayloadSize(sizeof(void*)) // XXX: find better value
    , NodeAllocator(new TFixedSizeAllocator(sizeof(TNode) + PayloadSize, alloc))
    , Flags(flags)
    , EntryCount(0)
    , NodeCount(1)
    , Packer(packer)
{
    Root = new (*NodeAllocator) TNode;
}

template <class T, class D, class S>
TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::~TCompactTrieBuilderImpl() {
    DestroyNode(Root);
}

template <class T, class D, class S>
void TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::ConvertSymbolArrayToChar(
                const TSymbol* key, size_t keylen, TTempBuf& buf, size_t buflen) const {
    char* ckeyptr = buf.Data();

    for (size_t i = 0; i < keylen; ++i) {
        TSymbol label = key[i];
        for (int j = (int)NCompactTrie::ExtraBits<TSymbol>(); j >= 0; j -= 8) {
            Y_ASSERT(ckeyptr < buf.Data() + buflen);
            *(ckeyptr++) = (char)(label >> j);
        }
    }

    buf.Proceed(buflen);
    Y_ASSERT(ckeyptr == buf.Data() + buf.Filled());
}

template <class T, class D, class S>
void TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::DestroyNode(TNode* thiz) {
    thiz->Subtree()->Destroy(this);
    NodeReleasePayload(thiz);
    thiz->~TNode();
    NodeAllocator->Release(thiz);
}

template <class T, class D, class S>
void TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::NodeReleasePayload(TNode* thiz) {
    switch (thiz->PayloadType) {
    case DATA_ABSENT:
    case DATA_INSIDE:
    case DATA_IN_MEMPOOL:
        break;
    case DATA_MALLOCED:
        delete[] thiz->PayloadAsPtr();
        break;
    default:
        abort();
    }
    thiz->PayloadType = DATA_ABSENT;
}

template <class T, class D, class S>
bool TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::AddEntry(
                const TSymbol* key, size_t keylen, const TData& value) {
    size_t datalen = Packer.MeasureLeaf(value);

    bool isNewAddition = false;
    char* place = AddEntryForData(key, keylen, datalen, isNewAddition);
    Packer.PackLeaf(place, value, datalen);
    return isNewAddition;
}

template <class T, class D, class S>
bool TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::AddEntryPtr(
                const TSymbol* key, size_t keylen, const char* value) {
    size_t datalen = Packer.SkipLeaf(value);

    bool isNewAddition = false;
    char* place = AddEntryForData(key, keylen, datalen, isNewAddition);
    memcpy(place, value, datalen);
    return isNewAddition;
}

template <class T, class D, class S>
bool TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::AddSubtreeInFile(
                const TSymbol* key, size_t keylen, const TString& fileName) {
    typedef typename TNode::ISubtree ISubtree;
    typedef typename TNode::TSubtreeInFile TSubtreeInFile;

    bool isNewAddition = false;
    TNode* node = AddEntryForSomething(key, keylen, isNewAddition);
    node->Subtree()->Destroy(this);
    node->Subtree()->~ISubtree();

    new (node->Subtree()) TSubtreeInFile(fileName);
    return isNewAddition;
}

template <class T, class D, class S>
bool TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::AddSubtreeInBuffer(
        const TSymbol* key, size_t keylen, TArrayWithSizeHolder<char>&& buffer) {

    typedef typename TNode::TBufferedSubtree TBufferedSubtree;

    bool isNewAddition = false;
    TNode* node = AddEntryForSomething(key, keylen, isNewAddition);
    node->Subtree()->Destroy(this);
    node->Subtree()->~ISubtree();

    auto subtree = new (node->Subtree()) TBufferedSubtree();
    subtree->Buffer.Swap(buffer);

    return isNewAddition;
}

template <class T, class D, class S>
typename TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::TNode*
                TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::AddEntryForSomething(
                                const TSymbol* key, size_t keylen, bool& isNewAddition) {
    using namespace NCompactTrie;

    EntryCount++;

    if (Flags & CTBF_VERBOSE)
        ShowProgress(EntryCount);

    TNode* current = Root;
    size_t passed;

    // Special case of empty key: replace it by 1-byte "\0" key.
    size_t ckeylen = keylen ? keylen * sizeof(TSymbol) : 1;
    TTempBuf ckeybuf(ckeylen);
    if (keylen == 0) {
        ckeybuf.Append("\0", 1);
    } else {
        ConvertSymbolArrayToChar(key, keylen, ckeybuf, ckeylen);
    }

    char* ckey = ckeybuf.Data();

    TNode* next;
    while ((ckeylen > 0) && (next = NodeForwardAdd(current, ckey, ckeylen, passed, &NodeCount)) != nullptr) {
        current = next;
        ckeylen -= passed;
        ckey += passed;
    }

    if (ckeylen != 0) {
        //new leaf
        NodeCount++;
        TNode* leaf = new (*NodeAllocator) TNode();
        NodeLinkTo(current, TBlob::Copy(ckey, ckeylen), leaf);
        current = leaf;
    }
    isNewAddition = (current->PayloadType == DATA_ABSENT);
    if ((Flags & CTBF_UNIQUE) && !isNewAddition)
        ythrow yexception() << "Duplicate key";
    return current;
}

template <class T, class D, class S>
char* TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::AddEntryForData(const TSymbol* key, size_t keylen,
        size_t datalen, bool& isNewAddition) {
    TNode* current = AddEntryForSomething(key, keylen, isNewAddition);
    NodeReleasePayload(current);
    if (datalen <= PayloadSize) {
        current->PayloadType = DATA_INSIDE;
    } else if (Flags & CTBF_PREFIX_GROUPED) {
        current->PayloadType = DATA_MALLOCED;
        current->PayloadAsPtr() = new char[datalen];
    } else {
        current->PayloadType = DATA_IN_MEMPOOL;
        current->PayloadAsPtr() = (char*) Pool.Allocate(datalen); // XXX: allocate unaligned
    }
    return current->GetPayload();
}

template <class T, class D, class S>
bool TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::FindEntry(const TSymbol* key, size_t keylen, TData* value) const {
    using namespace NCompactTrie;

    if (!keylen) {
        const char zero = '\0';
        return FindEntryImpl(&zero, 1, value);
    } else {
        size_t ckeylen = keylen * sizeof(TSymbol);
        TTempBuf ckeybuf(ckeylen);
        ConvertSymbolArrayToChar(key, keylen, ckeybuf, ckeylen);
        return FindEntryImpl(ckeybuf.Data(), ckeylen, value);
    }
}

template <class T, class D, class S>
bool TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::FindEntryImpl(const char* keyptr, size_t keylen, TData* value) const {
    const TNode* node = Root;
    bool result = false;
    TStringBuf key(keyptr, keylen);
    while (key && (node = node->Subtree()->Find(key, value, result, Packer))) {
    }
    return result;
}

template <class T, class D, class S>
bool TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::FindLongestPrefix(
                const TSymbol* key, size_t keylen, size_t* prefixlen, TData* value) const {
    using namespace NCompactTrie;

    if (!keylen) {
        const char zero = '\0';
        const bool ret = FindLongestPrefixImpl(&zero, 1, prefixlen, value);
        if (ret && prefixlen)
            *prefixlen = 0; // empty key found
        return ret;
    } else {
        size_t ckeylen = keylen * sizeof(TSymbol);
        TTempBuf ckeybuf(ckeylen);
        ConvertSymbolArrayToChar(key, keylen, ckeybuf, ckeylen);
        bool ret = FindLongestPrefixImpl(ckeybuf.Data(), ckeylen, prefixlen, value);
        if (ret && prefixlen && *prefixlen == 1 && ckeybuf.Data()[0] == '\0')
            *prefixlen = 0; // if we have found empty key, set prefixlen to zero
        else if (!ret) // try to find value with empty key, because empty key is prefix of a every key
            ret = FindLongestPrefix(nullptr, 0, prefixlen, value);

        if (ret && prefixlen)
            *prefixlen /= sizeof(TSymbol);

        return ret;
    }
}

template <class T, class D, class S>
bool TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::FindLongestPrefixImpl(const char* keyptr, size_t keylen, size_t* prefixLen, TData* value) const {
    const TNode* node = Root;
    const TNode* lastFinalNode = nullptr;
    bool endResult = false;
    TStringBuf key(keyptr, keylen);
    TStringBuf keyTail = key;
    TStringBuf lastFinalKeyTail;
    while (keyTail && (node = node->Subtree()->FindLongestPrefix(keyTail, value, endResult, Packer))) {
        if (endResult) // no more ways to find prefix and prefix has been found
            break;

        if (node->IsFinal()) {
            lastFinalNode = node;
            lastFinalKeyTail = keyTail;
        }
    }
    if (!endResult && lastFinalNode) {
        if (value)
            Packer.UnpackLeaf(lastFinalNode->GetPayload(), *value);
        keyTail = lastFinalKeyTail;
        endResult = true;
    }
    if (endResult && prefixLen)
        *prefixLen = keyTail ? key.size() - keyTail.size() : key.size();
    return endResult;
}

template <class T, class D, class S>
void TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::Clear() {
    DestroyNode(Root);
    Pool.Clear();
    NodeAllocator.Reset(new TFixedSizeAllocator(sizeof(TNode) + PayloadSize, TDefaultAllocator::Instance()));
    Root = new (*NodeAllocator) TNode;
    EntryCount = 0;
    NodeCount = 1;
}

template <class T, class D, class S>
size_t TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::Save(IOutputStream& os) const {
    const size_t len = NodeMeasureSubtree(Root);
    if (len != NodeSaveSubtree(Root, os))
        ythrow yexception() << "something wrong";

    return len;
}

template <class T, class D, class S>
size_t TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::SaveAndDestroy(IOutputStream& os) {
    const size_t len = NodeMeasureSubtree(Root);
    if (len != NodeSaveSubtreeAndDestroy(Root, os))
        ythrow yexception() << "something wrong";

    return len;
}

template <class T, class D, class S>
size_t TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::GetEntryCount() const {
    return EntryCount;
}

template <class T, class D, class S>
size_t TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::GetNodeCount() const {
    return NodeCount;
}

template <class T, class D, class S>
typename TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::TNode*
                TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::NodeForwardAdd(
                                TNode* thiz, const char* label, size_t len, size_t& passed, size_t* nodeCount) {
    typename TNode::TArcSet* arcSet = dynamic_cast<typename TNode::TArcSet*>(thiz->Subtree());
    if (!arcSet)
        ythrow yexception() << "Bad input order - expected input strings to be prefix-grouped.";

    typename TNode::TArcSet::iterator it = arcSet->Find(*label);

    if (it != arcSet->end()) {
        const char* arcLabel = it->Label.AsCharPtr();
        size_t arcLabelLen = it->Label.Length();

        for (passed = 0; (passed < len) && (passed < arcLabelLen) && (label[passed] == arcLabel[passed]); ++passed) {
            //just count
        }

        if (passed < arcLabelLen) {
            (*nodeCount)++;
            TNode* node = new (*NodeAllocator) TNode();
            NodeLinkTo(node, it->Label.SubBlob(passed, arcLabelLen), it->Node);

            it->Node = node;
            it->Label = it->Label.SubBlob(passed);
        }

        return it->Node;
    }

    return nullptr;
}

template <class T, class D, class S>
void TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::NodeLinkTo(TNode* thiz, const TBlob& label, TNode* node) {
    typename TNode::TArcSet* arcSet = dynamic_cast<typename TNode::TArcSet*>(thiz->Subtree());
    if (!arcSet)
        ythrow yexception() << "Bad input order - expected input strings to be prefix-grouped.";

    // Buffer the node at the last arc
    if ((Flags & CTBF_PREFIX_GROUPED) && !arcSet->empty())
        NodeBufferSubtree(arcSet->back().Node);

    arcSet->Add(label, node);
}

template <class T, class D, class S>
size_t TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::NodeMeasureSubtree(TNode* thiz) const {
    return (size_t)thiz->Subtree()->Measure(this);
}

template <class T, class D, class S>
ui64 TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::NodeSaveSubtree(TNode* thiz, IOutputStream& os) const {
    return thiz->Subtree()->Save(this, os);
}

template <class T, class D, class S>
ui64 TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::NodeSaveSubtreeAndDestroy(TNode* thiz, IOutputStream& os) {
    return thiz->Subtree()->SaveAndDestroy(this, os);
}

template <class T, class D, class S>
void TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::NodeBufferSubtree(TNode* thiz) {
    typedef typename TNode::TArcSet TArcSet;

    TArcSet* arcSet = dynamic_cast<TArcSet*>(thiz->Subtree());
    if (!arcSet)
        return;

    size_t bufferLength = (size_t)arcSet->Measure(this);
    TArrayWithSizeHolder<char> buffer;
    buffer.Resize(bufferLength);

    TMemoryOutput bufout(buffer.Get(), buffer.Size());

    ui64 written = arcSet->Save(this, bufout);
    Y_ASSERT(written == bufferLength);

    arcSet->Destroy(this);
    arcSet->~TArcSet();

    typename TNode::TBufferedSubtree* bufferedArcSet = new (thiz->Subtree()) typename TNode::TBufferedSubtree;

    bufferedArcSet->Buffer.Swap(buffer);
}

template <class T, class D, class S>
size_t TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::NodeMeasureLeafValue(TNode* thiz) const {
    if (!thiz->IsFinal())
        return 0;

    return Packer.SkipLeaf(thiz->GetPayload());
}

template <class T, class D, class S>
ui64 TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::NodeSaveLeafValue(TNode* thiz, IOutputStream& os) const {
    if (!thiz->IsFinal())
        return 0;

    size_t len = Packer.SkipLeaf(thiz->GetPayload());
    os.Write(thiz->GetPayload(), len);
    return len;
}

// TCompactTrieBuilder::TCompactTrieBuilderImpl::TNode::TArc

template <class T, class D, class S>
TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::TArc::TArc(const TBlob& lbl, TNode* nd)
    : Label(lbl)
    , Node(nd)
    , LeftOffset(0)
    , RightOffset(0)
{}

template <class T, class D, class S>
ui64 TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::ArcMeasure(
                const TArc* thiz, size_t leftsize, size_t rightsize) const {
    using namespace NCompactTrie;

    size_t coresize = 2 + NodeMeasureLeafValue(thiz->Node); // 2 == (char + flags)
    size_t treesize = NodeMeasureSubtree(thiz->Node);

    if (thiz->Label.Length() > 0)
        treesize += 2 * (thiz->Label.Length() - 1);

    // Triple measurements are needed because the space needed to store the offset
    // shall be added to the offset itself. Hence three iterations.
    size_t leftoffsetsize = leftsize ? MeasureOffset(coresize + treesize) : 0;
    size_t rightoffsetsize = rightsize ? MeasureOffset(coresize + treesize + leftsize) : 0;
    leftoffsetsize = leftsize ? MeasureOffset(coresize + treesize + leftoffsetsize + rightoffsetsize) : 0;
    rightoffsetsize = rightsize ? MeasureOffset(coresize + treesize + leftsize + leftoffsetsize + rightoffsetsize) : 0;
    leftoffsetsize = leftsize ? MeasureOffset(coresize + treesize + leftoffsetsize + rightoffsetsize) : 0;
    rightoffsetsize = rightsize ? MeasureOffset(coresize + treesize + leftsize + leftoffsetsize + rightoffsetsize) : 0;

    coresize += leftoffsetsize + rightoffsetsize;
    thiz->LeftOffset = leftsize ? coresize + treesize : 0;
    thiz->RightOffset = rightsize ? coresize + treesize + leftsize : 0;

    return coresize + treesize + leftsize + rightsize;
}

template <class T, class D, class S>
ui64 TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::ArcSaveSelf(const TArc* thiz, IOutputStream& os) const {
    using namespace NCompactTrie;

    ui64 written = 0;

    size_t leftoffsetsize = MeasureOffset(thiz->LeftOffset);
    size_t rightoffsetsize = MeasureOffset(thiz->RightOffset);

    size_t labelLen = thiz->Label.Length();

    for (size_t i = 0; i < labelLen; ++i) {
        char flags = 0;

        if (i == 0) {
            flags |= (leftoffsetsize << MT_LEFTSHIFT);
            flags |= (rightoffsetsize << MT_RIGHTSHIFT);
        }

        if (i == labelLen-1) {
            if (thiz->Node->IsFinal())
               flags |= MT_FINAL;

            if (!thiz->Node->IsLast())
                flags |= MT_NEXT;
        } else {
            flags |= MT_NEXT;
        }

        os.Write(&flags, 1);
        os.Write(&thiz->Label.AsCharPtr()[i], 1);
        written += 2;

        if (i == 0) {
            written += ArcSaveOffset(thiz->LeftOffset, os);
            written += ArcSaveOffset(thiz->RightOffset, os);
        }
    }

    written += NodeSaveLeafValue(thiz->Node, os);
    return written;
}

template <class T, class D, class S>
ui64 TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::ArcSave(const TArc* thiz, IOutputStream& os) const {
    ui64 written =  ArcSaveSelf(thiz, os);
    written += NodeSaveSubtree(thiz->Node, os);
    return written;
}

template <class T, class D, class S>
ui64 TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::ArcSaveAndDestroy(const TArc* thiz, IOutputStream& os) {
    ui64 written = ArcSaveSelf(thiz, os);
    written += NodeSaveSubtreeAndDestroy(thiz->Node, os);
    return written;
}

// TCompactTrieBuilder::TCompactTrieBuilderImpl::TNode::TArcSet

template <class T, class D, class S>
typename TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::TNode::TArcSet::iterator
                    TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::TNode::TArcSet::Find(char ch) {
    using namespace NCompTriePrivate;
    iterator it = LowerBound(this->begin(), this->end(), ch, TCmp());

    if (it != this->end() && it->Label[0] == (unsigned char)ch) {
        return it;
    }

    return this->end();
}

template <class T, class D, class S>
typename TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::TNode::TArcSet::const_iterator
                    TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::TNode::TArcSet::Find(char ch) const {
    using namespace NCompTriePrivate;
    const_iterator it = LowerBound(this->begin(), this->end(), ch, TCmp());

    if (it != this->end() && it->Label[0] == (unsigned char)ch) {
        return it;
    }

    return this->end();
}

template <class T, class D, class S>
void TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::TNode::TArcSet::Add(const TBlob& s, TNode* node) {
    using namespace NCompTriePrivate;
    this->insert(LowerBound(this->begin(), this->end(), s[0], TCmp()), TArc(s, node));
}

template <class T, class D, class S>
const typename TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::TNode*
    TCompactTrieBuilder<T, D, S>::TCompactTrieBuilderImpl::TNode::TArcSet::Find(
                                TStringBuf& key, TData* value, bool& result, const TPacker& packer) const {
    result = false;
    if (!key)
        return nullptr;

    const const_iterator it = Find(key[0]);
    if (it != this->end()) {
        const char* const arcLabel = it->Label.AsCharPtr();
        const size_t arcLabelLen = it->Label.Length();
        if (key.size() >= arcLabelLen && memcmp(key.data(), arcLabel, arcLabelLen) == 0) {
            const TStringBuf srcKey = key;
            key = key.SubStr(arcLabelLen);
            const TNode* const node = it->Node;
            if (srcKey.size() == arcLabelLen) {
                // unpack value of it->Node, if it has value
                if (!node->IsFinal())
                    return nullptr;

                if (value)
                    packer.UnpackLeaf(node->GetPayload(), *value);

                result = true;
                return nullptr;
            }

            // find in subtree
            return node;
        }
    }

    return nullptr;
}

// Different

//----------------------------------------------------------------------------------------------------------------------
// Minimize the trie. The result is equivalent to the original
// trie, except that it takes less space (and has marginally lower
// performance, because of eventual epsilon links).
// The algorithm is as follows: starting from the largest pieces, we find
// nodes that have identical continuations  (Daciuk's right language),
// and repack the trie. Repacking is done in-place, so memory is less
// of an issue; however, it may take considerable time.

// IMPORTANT: never try to reminimize an already minimized trie or a trie with fast layout.
// Because of non-local structure and epsilon links, it won't work
// as you expect it to, and can destroy the trie in the making.

template <class TPacker>
size_t CompactTrieMinimize(IOutputStream& os, const char* data, size_t datalength, bool verbose /*= false*/, const TPacker& packer /*= TPacker()*/, NCompactTrie::EMinimizeMode mode) {
    using namespace NCompactTrie;
    return CompactTrieMinimizeImpl(os, data, datalength, verbose, &packer, mode);
}

template <class TTrieBuilder>
size_t CompactTrieMinimize(IOutputStream& os, const TTrieBuilder& builder, bool verbose /*=false*/) {
    TBufferStream buftmp;
    size_t len = builder.Save(buftmp);
    return CompactTrieMinimize<typename TTrieBuilder::TPacker>(os, buftmp.Buffer().Data(), len, verbose);
}

//----------------------------------------------------------------------------------------------------------------
// Lay the trie in memory in such a way that there are less cache misses when jumping from root to leaf.
// The trie becomes about 2% larger, but the access became about 25% faster in our experiments.
// Can be called on minimized and non-minimized tries, in the first case in requires half a trie more memory.
// Calling it the second time on the same trie does nothing.
//
// The algorithm is based on van Emde Boas layout as described in the yandex data school lectures on external memory algoritms
// by Maxim Babenko and Ivan Puzyrevsky. The difference is that when we cut the tree into levels
// two nodes connected by a forward link are put into the same level (because they usually lie near each other in the original tree).
// The original paper (describing the layout in Section 2.1) is:
// Michael A. Bender, Erik D. Demaine, Martin Farach-Colton. Cache-Oblivious B-Trees
//      SIAM Journal on Computing, volume 35, number 2, 2005, pages 341-358.
// Available on the web: http://erikdemaine.org/papers/CacheObliviousBTrees_SICOMP/
// Or: Michael A. Bender, Erik D. Demaine, and Martin Farach-Colton. Cache-Oblivious B-Trees
//      Proceedings of the 41st Annual Symposium
//      on Foundations of Computer Science (FOCS 2000), Redondo Beach, California, November 12-14, 2000, pages 399-409.
// Available on the web: http://erikdemaine.org/papers/FOCS2000b/
// (there is not much difference between these papers, actually).
//
template <class TPacker>
size_t CompactTrieMakeFastLayout(IOutputStream& os, const char* data, size_t datalength, bool verbose /*= false*/, const TPacker& packer /*= TPacker()*/) {
    using namespace NCompactTrie;
    return CompactTrieMakeFastLayoutImpl(os, data, datalength, verbose, &packer);
}

template <class TTrieBuilder>
size_t CompactTrieMakeFastLayout(IOutputStream& os, const TTrieBuilder& builder, bool verbose /*=false*/) {
    TBufferStream buftmp;
    size_t len = builder.Save(buftmp);
    return CompactTrieMakeFastLayout<typename TTrieBuilder::TPacker>(os, buftmp.Buffer().Data(), len, verbose);
}

template <class TPacker>
size_t CompactTrieMinimizeAndMakeFastLayout(IOutputStream& os, const char* data, size_t datalength, bool verbose/*=false*/, const TPacker& packer/*= TPacker()*/) {
    TBufferStream buftmp;
    size_t len = CompactTrieMinimize(buftmp, data, datalength, verbose, packer);
    return CompactTrieMakeFastLayout(os, buftmp.Buffer().Data(), len, verbose, packer);
}

template <class TTrieBuilder>
size_t CompactTrieMinimizeAndMakeFastLayout(IOutputStream& os, const TTrieBuilder& builder, bool verbose /*=false*/) {
    TBufferStream buftmp;
    size_t len = CompactTrieMinimize(buftmp, builder, verbose);
    return CompactTrieMakeFastLayout<typename TTrieBuilder::TPacker>(os, buftmp.Buffer().Data(), len, verbose);
}

