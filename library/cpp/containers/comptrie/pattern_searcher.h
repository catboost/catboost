#pragma once

#include "comptrie_builder.h"
#include "comptrie_trie.h"
#include "comptrie_impl.h"
#include <library/cpp/packers/packers.h>

#include <util/system/yassert.h>
#include <util/generic/vector.h>
#include <util/generic/deque.h>
#include <util/stream/str.h>

// Aho-Corasick algorithm implementation using CompactTrie implementation of Sedgewick's T-trie

namespace NCompactTrie {
    struct TSuffixLink {
        ui64 NextSuffixOffset;
        ui64 NextSuffixWithDataOffset;

        TSuffixLink(ui64 nextSuffixOffset = 0, ui64 nextSuffixWithDataOffset = 0)
            : NextSuffixOffset(nextSuffixOffset)
            , NextSuffixWithDataOffset(nextSuffixWithDataOffset)
        {
        }
    };

    const size_t FLAGS_SIZE = sizeof(char);
    const size_t SYMBOL_SIZE = sizeof(char);
}

template <class T = char, class D = ui64, class S = TCompactTriePacker<D>>
class TCompactPatternSearcherBuilder : protected TCompactTrieBuilder<T, D, S> {
public:
    typedef T TSymbol;
    typedef D TData;
    typedef S TPacker;

    typedef typename TCompactTrieKeySelector<TSymbol>::TKey TKey;
    typedef typename TCompactTrieKeySelector<TSymbol>::TKeyBuf TKeyBuf;

    typedef TCompactTrieBuilder<T, D, S> TBase;

public:
    TCompactPatternSearcherBuilder() {
        TBase::Impl = MakeHolder<TCompactPatternSearcherBuilderImpl>();
    }

    bool Add(const TSymbol* key, size_t keyLength, const TData& value) {
        return TBase::Impl->AddEntry(key, keyLength, value);
    }
    bool Add(const TKeyBuf& key, const TData& value) {
        return Add(key.data(), key.size(), value);
    }

    bool Find(const TSymbol* key, size_t keyLength, TData* value) const {
        return TBase::Impl->FindEntry(key, keyLength, value);
    }
    bool Find(const TKeyBuf& key, TData* value = nullptr) const {
        return Find(key.data(), key.size(), value);
    }

    size_t Save(IOutputStream& os) const {
        size_t trieSize = TBase::Impl->MeasureByteSize();
        TBufferOutput serializedTrie(trieSize);
        TBase::Impl->Save(serializedTrie);

        auto serializedTrieBuffer = serializedTrie.Buffer();
        CalculateSuffixLinks(
            serializedTrieBuffer.Data(),
            serializedTrieBuffer.Data() + serializedTrieBuffer.Size()
        );

        os.Write(serializedTrieBuffer.Data(), serializedTrieBuffer.Size());
        return trieSize;
    }

    TBlob Save() const {
        TBufferStream buffer;
        Save(buffer);
        return TBlob::FromStream(buffer);
    }

    size_t SaveToFile(const TString& fileName) const {
        TFileOutput out(fileName);
        return Save(out);
    }

    size_t MeasureByteSize() const {
        return TBase::Impl->MeasureByteSize();
    }

private:
    void CalculateSuffixLinks(char* trieStart, const char* trieEnd) const;

protected:
    class TCompactPatternSearcherBuilderImpl : public TBase::TCompactTrieBuilderImpl {
    public:
        typedef typename TBase::TCompactTrieBuilderImpl TImplBase;

        TCompactPatternSearcherBuilderImpl(
            TCompactTrieBuilderFlags flags = CTBF_NONE,
            TPacker packer = TPacker(),
            IAllocator* alloc = TDefaultAllocator::Instance()
        ) : TImplBase(flags, packer, alloc) {
        }

        ui64 ArcMeasure(
            const typename TImplBase::TArc* arc,
            size_t leftSize,
            size_t rightSize
        ) const override {
            using namespace NCompactTrie;

            size_t coreSize = SYMBOL_SIZE + FLAGS_SIZE +
                sizeof(TSuffixLink) +
                this->NodeMeasureLeafValue(arc->Node);
            size_t treeSize = this->NodeMeasureSubtree(arc->Node);

            if (arc->Label.Length() > 0)
                treeSize += (SYMBOL_SIZE + FLAGS_SIZE + sizeof(TSuffixLink)) *
                    (arc->Label.Length() - 1);

            // Triple measurements are needed because the space needed to store the offset
            // shall be added to the offset itself. Hence three iterations.
            size_t leftOffsetSize = 0;
            size_t rightOffsetSize = 0;
            for (size_t iteration = 0; iteration < 3; ++iteration) {
                leftOffsetSize = leftSize ? MeasureOffset(
                    coreSize + treeSize + leftOffsetSize + rightOffsetSize) : 0;
                rightOffsetSize = rightSize ? MeasureOffset(
                    coreSize + treeSize + leftSize + leftOffsetSize + rightOffsetSize) : 0;
            }

            coreSize += leftOffsetSize + rightOffsetSize;
            arc->LeftOffset = leftSize ? coreSize + treeSize : 0;
            arc->RightOffset = rightSize ? coreSize + treeSize + leftSize : 0;

            return coreSize + treeSize + leftSize + rightSize;
        }

        ui64 ArcSaveSelf(const typename TImplBase::TArc* arc, IOutputStream& os) const override {
            using namespace NCompactTrie;

            ui64 written = 0;

            size_t leftOffsetSize = MeasureOffset(arc->LeftOffset);
            size_t rightOffsetSize = MeasureOffset(arc->RightOffset);

            size_t labelLen = arc->Label.Length();

            for (size_t labelPos = 0; labelPos < labelLen; ++labelPos) {
                char flags = 0;

                if (labelPos == 0) {
                    flags |= (leftOffsetSize << MT_LEFTSHIFT);
                    flags |= (rightOffsetSize << MT_RIGHTSHIFT);
                }

                if (labelPos == labelLen - 1) {
                    if (arc->Node->IsFinal())
                        flags |= MT_FINAL;
                    if (!arc->Node->IsLast())
                        flags |= MT_NEXT;
                } else {
                    flags |= MT_NEXT;
                }

                os.Write(&flags, 1);
                os.Write(&arc->Label.AsCharPtr()[labelPos], 1);
                written += 2;

                TSuffixLink suffixlink;
                os.Write(&suffixlink, sizeof(TSuffixLink));
                written += sizeof(TSuffixLink);

                if (labelPos == 0) {
                    written += ArcSaveOffset(arc->LeftOffset, os);
                    written += ArcSaveOffset(arc->RightOffset, os);
                }
            }

            written += this->NodeSaveLeafValue(arc->Node, os);
            return written;
        }
    };
};


template <class T>
struct TPatternMatch {
    ui64 End;
    T Data;

    TPatternMatch(ui64 end, const T& data)
        : End(end)
        , Data(data)
    {
    }
};


template <class T = char, class D = ui64, class S = TCompactTriePacker<D>>
class TCompactPatternSearcher {
public:
    typedef T TSymbol;
    typedef D TData;
    typedef S TPacker;

    typedef typename TCompactTrieKeySelector<TSymbol>::TKey TKey;
    typedef typename TCompactTrieKeySelector<TSymbol>::TKeyBuf TKeyBuf;

    typedef TCompactTrie<TSymbol, TData, TPacker> TTrie;
public:
    TCompactPatternSearcher()
    {
    }

    explicit TCompactPatternSearcher(const TBlob& data)
        : Trie(data)
    {
    }

    TCompactPatternSearcher(const char* data, size_t size)
        : Trie(data, size)
    {
    }

    TVector<TPatternMatch<TData>> SearchMatches(const TSymbol* text, size_t textSize) const;
    TVector<TPatternMatch<TData>> SearchMatches(const TKeyBuf& text) const {
        return SearchMatches(text.data(), text.size());
    }
private:
    TTrie Trie;
};

////////////////////
// Implementation //
////////////////////

namespace {

template <class TData, class TPacker>
char ReadNode(
    char* nodeStart,
    char*& leftSibling,
    char*& rightSibling,
    char*& directChild,
    NCompactTrie::TSuffixLink*& suffixLink,
    TPacker packer = TPacker()
) {
    char* dataPos = nodeStart;
    char flags = *(dataPos++);

    Y_ASSERT(!NCompactTrie::IsEpsilonLink(flags)); // Epsilon links are not allowed

    char label = *(dataPos++);

    suffixLink = (NCompactTrie::TSuffixLink*)dataPos;
    dataPos += sizeof(NCompactTrie::TSuffixLink);

    { // Left branch
        size_t offsetLength = NCompactTrie::LeftOffsetLen(flags);
        size_t leftOffset = NCompactTrie::UnpackOffset(dataPos, offsetLength);
        leftSibling = leftOffset ? (nodeStart + leftOffset) : nullptr;

        dataPos += offsetLength;
    }


    { // Right branch
        size_t offsetLength = NCompactTrie::RightOffsetLen(flags);
        size_t rightOffset = NCompactTrie::UnpackOffset(dataPos, offsetLength);
        rightSibling = rightOffset ? (nodeStart + rightOffset) : nullptr;

        dataPos += offsetLength;
    }

    directChild = nullptr;
    if (flags & NCompactTrie::MT_NEXT) {
        directChild = dataPos;
        if (flags & NCompactTrie::MT_FINAL) {
            directChild += packer.SkipLeaf(directChild);
        }
    }

    return label;
}

template <class TData, class TPacker>
char ReadNodeConst(
    const char* nodeStart,
    const char*& leftSibling,
    const char*& rightSibling,
    const char*& directChild,
    const char*& data,
    NCompactTrie::TSuffixLink& suffixLink,
    TPacker packer = TPacker()
) {
    const char* dataPos = nodeStart;
    char flags = *(dataPos++);

    Y_ASSERT(!NCompactTrie::IsEpsilonLink(flags)); // Epsilon links are not allowed

    char label = *(dataPos++);

    suffixLink = *((NCompactTrie::TSuffixLink*)dataPos);
    dataPos += sizeof(NCompactTrie::TSuffixLink);

    { // Left branch
        size_t offsetLength = NCompactTrie::LeftOffsetLen(flags);
        size_t leftOffset = NCompactTrie::UnpackOffset(dataPos, offsetLength);
        leftSibling = leftOffset ? (nodeStart + leftOffset) : nullptr;

        dataPos += offsetLength;
    }


    { // Right branch
        size_t offsetLength = NCompactTrie::RightOffsetLen(flags);
        size_t rightOffset = NCompactTrie::UnpackOffset(dataPos, offsetLength);
        rightSibling = rightOffset ? (nodeStart + rightOffset) : nullptr;

        dataPos += offsetLength;
    }

    data = nullptr;
    if (flags & NCompactTrie::MT_FINAL) {
        data = dataPos;
    }
    directChild = nullptr;
    if (flags & NCompactTrie::MT_NEXT) {
        directChild = dataPos;
        if (flags & NCompactTrie::MT_FINAL) {
            directChild += packer.SkipLeaf(directChild);
        }
    }

    return label;
}

Y_FORCE_INLINE bool Advance(
    const char*& dataPos,
    const char* const dataEnd,
    char label
) {
    if (dataPos == nullptr) {
        return false;
    }

    while (dataPos < dataEnd) {
        size_t offsetLength, offset;
        const char* startPos = dataPos;
        char flags = *(dataPos++);
        char symbol = *(dataPos++);
        dataPos += sizeof(NCompactTrie::TSuffixLink);

        // Left branch
        offsetLength = NCompactTrie::LeftOffsetLen(flags);
        if ((unsigned char)label < (unsigned char)symbol) {
            offset = NCompactTrie::UnpackOffset(dataPos, offsetLength);
            if (!offset)
                break;

            dataPos = startPos + offset;
            continue;
        }

        dataPos += offsetLength;

        // Right branch
        offsetLength = NCompactTrie::RightOffsetLen(flags);
        if ((unsigned char)label > (unsigned char)symbol) {
            offset = NCompactTrie::UnpackOffset(dataPos, offsetLength);
            if (!offset)
                break;

            dataPos = startPos + offset;
            continue;
        }

        dataPos = startPos;
        return true;
    }

    // if we got here, we're past the dataend - bail out ASAP
    dataPos = nullptr;
    return false;
}

} // anonymous

template <class T, class D, class S>
void TCompactPatternSearcherBuilder<T, D, S>::CalculateSuffixLinks(
    char* trieStart,
    const char* trieEnd
) const {
    struct TBfsElement {
        char* Node;
        const char* Parent;

        TBfsElement(char* node, const char* parent)
            : Node(node)
            , Parent(parent)
        {
        }
    };

    TDeque<TBfsElement> bfsQueue;
    if (trieStart && trieStart != trieEnd) {
        bfsQueue.emplace_back(trieStart, nullptr);
    }

    while (!bfsQueue.empty()) {
        auto front = bfsQueue.front();
        char* node = front.Node;
        const char* parent = front.Parent;
        bfsQueue.pop_front();

        char* leftSibling;
        char* rightSibling;
        char* directChild;
        NCompactTrie::TSuffixLink* suffixLink;

        char label = ReadNode<TData, TPacker>(
            node,
            leftSibling,
            rightSibling,
            directChild,
            suffixLink
        );

        const char* suffix;

        if (parent == nullptr) {
            suffix = node;
        } else {
            const char* parentOfSuffix = parent;
            const char* temp;
            do {
                NCompactTrie::TSuffixLink parentOfSuffixSuffixLink;

                ReadNodeConst<TData, TPacker>(
                    parentOfSuffix,
                    /*left*/temp,
                    /*right*/temp,
                    /*direct*/temp,
                    /*data*/temp,
                    parentOfSuffixSuffixLink
                );
                if (parentOfSuffixSuffixLink.NextSuffixOffset == 0) {
                    suffix = trieStart;
                    if (!Advance(suffix, trieEnd, label)) {
                        suffix = node;
                    }
                    break;
                }
                parentOfSuffix += parentOfSuffixSuffixLink.NextSuffixOffset;

                NCompactTrie::TSuffixLink tempSuffixLink;
                ReadNodeConst<TData, TPacker>(
                    parentOfSuffix,
                    /*left*/temp,
                    /*right*/temp,
                    /*direct*/suffix,
                    /*data*/temp,
                    tempSuffixLink
                );

                if (suffix == nullptr) {
                    continue;
                }
            } while (!Advance(suffix, trieEnd, label));
        }

        suffixLink->NextSuffixOffset = suffix - node;

        NCompactTrie::TSuffixLink suffixSuffixLink;
        const char* suffixData;
        const char* temp;
        ReadNodeConst<TData, TPacker>(
            suffix,
            /*left*/temp,
            /*right*/temp,
            /*direct*/temp,
            suffixData,
            suffixSuffixLink
        );
        suffixLink->NextSuffixWithDataOffset = suffix - node;
        if (suffixData == nullptr) {
            suffixLink->NextSuffixWithDataOffset += suffixSuffixLink.NextSuffixWithDataOffset;
        }

        if (directChild) {
            bfsQueue.emplace_back(directChild, node);
        }

        if (leftSibling) {
            bfsQueue.emplace_front(leftSibling, parent);
        }

        if (rightSibling) {
            bfsQueue.emplace_front(rightSibling, parent);
        }
    }
}


template<class T, class D, class S>
TVector<TPatternMatch<D>> TCompactPatternSearcher<T, D, S>::SearchMatches(
    const TSymbol* text,
    size_t textSize
) const {
    const char* temp;
    NCompactTrie::TSuffixLink tempSuffixLink;

    const auto& trieData = Trie.Data();
    const char* trieStart = trieData.AsCharPtr();
    size_t dataSize = trieData.Length();
    const char* trieEnd = trieStart + dataSize;

    const char* lastNode = nullptr;
    const char* currentSubtree = trieStart;

    TVector<TPatternMatch<TData>> matches;

    for (const TSymbol* position = text; position < text + textSize; ++position) {
        TSymbol symbol = *position;
        for (i64 i = (i64)NCompactTrie::ExtraBits<TSymbol>(); i >= 0; i -= 8) {
            char label = (char)(symbol >> i);

            // Find first suffix extendable by label
            while (true) {
                const char* nextLastNode = currentSubtree;
                if (Advance(nextLastNode, trieEnd, label)) {
                    lastNode = nextLastNode;
                    ReadNodeConst<TData, TPacker>(
                        lastNode,
                        /*left*/temp,
                        /*right*/temp,
                        currentSubtree,
                        /*data*/temp,
                        tempSuffixLink
                    );
                    break;
                } else {
                    if (lastNode == nullptr) {
                        break;
                    }
                }

                NCompactTrie::TSuffixLink suffixLink;
                ReadNodeConst<TData, TPacker>(
                    lastNode,
                    /*left*/temp,
                    /*right*/temp,
                    /*direct*/temp,
                    /*data*/temp,
                    suffixLink
                );
                if (suffixLink.NextSuffixOffset == 0) {
                    lastNode = nullptr;
                    currentSubtree = trieStart;
                    continue;
                }
                lastNode += suffixLink.NextSuffixOffset;
                ReadNodeConst<TData, TPacker>(
                    lastNode,
                    /*left*/temp,
                    /*right*/temp,
                    currentSubtree,
                    /*data*/temp,
                    tempSuffixLink
                );
            }

            // Iterate through all suffixes
            const char* suffix = lastNode;
            while (suffix != nullptr) {
                const char* nodeData;
                NCompactTrie::TSuffixLink suffixLink;
                ReadNodeConst<TData, TPacker>(
                    suffix,
                    /*left*/temp,
                    /*right*/temp,
                    /*direct*/temp,
                    nodeData,
                    suffixLink
                );
                if (nodeData != nullptr) {
                    TData data;
                    Trie.GetPacker().UnpackLeaf(nodeData, data);
                    matches.emplace_back(
                        position - text,
                        data
                    );
                }
                if (suffixLink.NextSuffixOffset == 0) {
                    break;
                }
                suffix += suffixLink.NextSuffixWithDataOffset;
            }
        }
    }

    return matches;
}
