#include "codecs.h"
#include "common.h"
#include "register.h"

#include <util/stream/null.h>
#include <util/string/join.h>
#include <util/system/align.h>
#include <util/system/unaligned_mem.h>
#include <util/generic/hash.h>
#include <util/generic/deque.h>
#include <util/generic/buffer.h>
#include <util/generic/array_ref.h>
#include <util/generic/singleton.h>
#include <util/generic/algorithm.h>
#include <util/generic/mem_copy.h>

using namespace NBlockCodecs;

namespace {

    struct TCodecFactory {
        inline TCodecFactory() {
            Add(&Null);
        }

        inline const ICodec* Find(const TStringBuf& name) const {
            auto it = Registry.find(name);

            if (it == Registry.end()) {
                ythrow TNotFound() << "can not found " << name << " codec";
            }

            return it->second;
        }

        inline void ListCodecs(TCodecList& lst) const {
            for (const auto& it : Registry) {
                lst.push_back(it.first);
            }

            Sort(lst.begin(), lst.end());
        }

        inline void Add(ICodec* codec) {
            Registry[codec->Name()] = codec;
        }

        inline void Add(TCodecPtr codec) {
            Codecs.push_back(std::move(codec));
            Add(Codecs.back().Get());
        }

        inline void Alias(TStringBuf from, TStringBuf to) {
            Tmp.emplace_back(from);
            Registry[Tmp.back()] = Registry[to];
        }

        TDeque<TString> Tmp;
        TNullCodec Null;
        TVector<TCodecPtr> Codecs;
        typedef THashMap<TStringBuf, ICodec*> TRegistry;
        TRegistry Registry;

        // SEARCH-8344: Global decompressed size limiter (to prevent remote DoS)
        size_t MaxPossibleDecompressedLength = Max<size_t>();
    };
}

const ICodec* NBlockCodecs::Codec(const TStringBuf& name) {
    return Singleton<TCodecFactory>()->Find(name);
}

TCodecList NBlockCodecs::ListAllCodecs() {
    TCodecList ret;

    Singleton<TCodecFactory>()->ListCodecs(ret);

    return ret;
}

TString NBlockCodecs::ListAllCodecsAsString() {
    return JoinSeq(TStringBuf(","), ListAllCodecs());
}

void NBlockCodecs::RegisterCodec(TCodecPtr codec) {
    Singleton<TCodecFactory>()->Add(std::move(codec));
}

void NBlockCodecs::RegisterAlias(TStringBuf from, TStringBuf to) {
    Singleton<TCodecFactory>()->Alias(from, to);
}

void NBlockCodecs::SetMaxPossibleDecompressedLength(size_t maxPossibleDecompressedLength) {
    Singleton<TCodecFactory>()->MaxPossibleDecompressedLength = maxPossibleDecompressedLength;
}

size_t NBlockCodecs::GetMaxPossibleDecompressedLength() {
    return Singleton<TCodecFactory>()->MaxPossibleDecompressedLength;
}

size_t ICodec::GetDecompressedLength(const TData& in) const {
    const size_t len = DecompressedLength(in);

    Y_ENSURE(
        len <= NBlockCodecs::GetMaxPossibleDecompressedLength(),
        "Attempt to decompress the block that is larger than maximum possible decompressed length, "
        "see SEARCH-8344 for details. "
    );
    return len;
}

void ICodec::Encode(const TData& in, TBuffer& out) const {
    const size_t maxLen = MaxCompressedLength(in);

    out.Reserve(maxLen);
    out.Resize(Compress(in, out.Data()));
}

void ICodec::Decode(const TData& in, TBuffer& out) const {
    const size_t len = GetDecompressedLength(in);

    out.Reserve(len);
    out.Resize(Decompress(in, out.Data()));
}

void ICodec::Encode(const TData& in, TString& out) const {
    const size_t maxLen = MaxCompressedLength(in);
    out.ReserveAndResize(maxLen);

    size_t actualLen = Compress(in, out.begin());
    Y_ASSERT(actualLen <= maxLen);
    out.resize(actualLen);
}

void ICodec::Decode(const TData& in, TString& out) const {
    const size_t maxLen = GetDecompressedLength(in);
    out.ReserveAndResize(maxLen);

    size_t actualLen = Decompress(in, out.begin());
    Y_ASSERT(actualLen <= maxLen);
    out.resize(actualLen);
}

ICodec::~ICodec() = default;
