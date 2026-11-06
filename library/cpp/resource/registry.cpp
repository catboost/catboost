#include "registry.h"

#include "resource.h"
#include "static_hash.h"

#include <library/cpp/containers/absl/flat_hash_map.h>
#include <library/cpp/blockcodecs/core/codecs.h>

#include <util/system/yassert.h>
#include <util/generic/hash.h>
#include <util/generic/deque.h>
#include <util/generic/singleton.h>
#include <util/generic/yexception.h>
#include <util/system/env.h>

using namespace NResource;
using namespace NBlockCodecs;

// Static resource table assembled at link time by library/python/runtime_py3/pyres_ld_plugin.py.
// Binaries without packed python resources get the empty one from static_blob_default.cpp.
extern "C" const char __pyres_static_blob[];

namespace {
    inline const ICodec* GetCodec() noexcept {
        static const ICodec* ret = Codec("zstd08_5");

        return ret;
    }

    inline const NStaticHash::TTable* StaticTable() noexcept {
        return NStaticHash::TTable::FromBlob(__pyres_static_blob);
    }

    inline TStringBuf AsBuf(std::string_view s) noexcept {
        return TStringBuf(s.data(), s.size());
    }

    typedef std::pair<TStringBuf, TStringBuf> TDescriptor;

    [[noreturn]]
    void ReportRedefinitionError(const TStringBuf key, const TStringBuf data, const ui64 kk, const TDescriptor& prev) noexcept {
        const auto& [prevKey, value] = prev;
        Y_ABORT_UNLESS(key == prevKey, "Internal hash collision:"
                 " old key: %s,"
                 " new key: %s,"
                 " hash: %" PRIx64 ".",
                 TString{prevKey}.Quote().c_str(),
                 TString{key}.Quote().c_str(),
                 kk);
        size_t vsize = GetCodec()->DecompressedLength(value);
        size_t dsize = GetCodec()->DecompressedLength(data);
        if (vsize + dsize < 1000) {
            Y_ABORT_UNLESS(false, "Redefinition of key %s:\n"
                     "  old value: %s,\n"
                     "  new value: %s.",
                     TString{key}.Quote().c_str(),
                     Decompress(value).Quote().c_str(),
                     Decompress(data).Quote().c_str());
        } else {
            Y_ABORT_UNLESS(false, "Redefinition of key %s,"
                     " old size: %zu,"
                     " new size: %zu.",
                     TString{key}.Quote().c_str(), vsize, dsize);
        }
    }

    TString DecompressDiag(const TStringBuf key, const TStringBuf data) {
        // temporary
        // https://st.yandex-team.ru/DEVTOOLS-3985
        try {
            return Decompress(data);
        } catch (const yexception& e) {
            if (GetEnv("RESOURCE_DECOMPRESS_DIAG")) {
                Cerr << "Can't decompress resource " << key << Endl << e.what() << Endl;
            }
            throw e;
        }
    }

    struct TStore final: public IStore, public absl::flat_hash_map<ui64, TDescriptor*> {
        static inline ui64 ToK(TStringBuf k) {
            return NHashPrivate::ComputeStringHash(k.data(), k.size());
        }

        bool FindStatic(ui64 kk, TStringBuf key, NStaticHash::TRecord* out) const noexcept {
            return S_ && S_->FindByHash(kk, std::string_view(key.data(), key.size()), out);
        }

        void Store(const TStringBuf key, const TStringBuf data) override {
            auto kk = ToK(key);

            if (NStaticHash::TRecord rec; FindStatic(kk, key, &rec)) {
                if (AsBuf(rec.Value) != data) {
                    ReportRedefinitionError(key, data, kk, TDescriptor(AsBuf(rec.Key), AsBuf(rec.Value)));
                    Y_UNREACHABLE();
                }
                return; // already served by the static table
            }

            const auto [it, unique] = try_emplace(kk, nullptr);
            if (!unique) {
                const auto& [_, value] = *it->second;
                if (value != data) {
                    ReportRedefinitionError(key, data, kk, *it->second);
                    Y_UNREACHABLE();
                }
            } else {
                D_.push_back(TDescriptor(key, data));
                it->second = &D_.back();
            }

            Y_ABORT_UNLESS(size() == D_.size(), "size mismatch");
        }

        bool Has(const TStringBuf key) const override {
            const auto kk = ToK(key);
            NStaticHash::TRecord rec;
            return contains(kk) || FindStatic(kk, key, &rec);
        }

        bool FindExact(const TStringBuf key, TString* out) const override {
            const auto kk = ToK(key);
            if (auto res = find(kk); res != end()) {
                *out = DecompressDiag(key, res->second->second);
                return true;
            }
            if (NStaticHash::TRecord rec; FindStatic(kk, key, &rec)) {
                *out = DecompressDiag(key, AsBuf(rec.Value));
                return true;
            }

            return false;
        }

        void FindMatch(const TStringBuf subkey, IMatch& cb) const override {
            for (const auto& it : D_) {
                if (it.first.StartsWith(subkey)) {
                    const TResource res = {it.first, DecompressDiag(it.first, it.second)};
                    cb.OnMatch(res);
                }
            }
            if (S_) {
                // TODO: records are sorted by key, the prefix range can be found by binary search
                for (size_t i = 0, n = S_->Count(); i < n; ++i) {
                    const auto rec = S_->ByIndex(i);
                    const TStringBuf key = AsBuf(rec.Key);
                    if (key.StartsWith(subkey)) {
                        const TResource res = {key, DecompressDiag(key, AsBuf(rec.Value))};
                        cb.OnMatch(res);
                    }
                }
            }
        }

        size_t Count() const noexcept override {
            return D_.size() + (S_ ? S_->Count() : 0);
        }

        TStringBuf KeyByIndex(size_t idx) const override {
            if (idx < D_.size()) {
                return D_.at(idx).first;
            }
            idx -= D_.size();
            Y_ENSURE(S_ && idx < S_->Count(), "resource index out of range");
            return AsBuf(S_->ByIndex(idx).Key);
        }

        typedef TDeque<TDescriptor> TDescriptors;
        TDescriptors D_;
        const NStaticHash::TTable* S_ = StaticTable();
    };
}

TString NResource::Compress(const TStringBuf data) {
    return GetCodec()->Encode(data);
}

TString NResource::Decompress(const TStringBuf data) {
    return GetCodec()->Decode(data);
}

IStore* NResource::CommonStore() {
    return SingletonWithPriority<TStore, 0>();
}

NResource::TRegHelper::TRegHelper(const TStringBuf key, const TStringBuf data) {
    CommonStore()->Store(key, data);
}

int NResource::LightRegisterI(const char* key, const char* data, const char* data_end) {
    CommonStore()->Store(TStringBuf(key), TStringBuf(data, data_end));
    return 0;

}
int NResource::LightRegisterS(const char* key, const char* data, unsigned long data_len) {
    CommonStore()->Store(TStringBuf(key), TStringBuf(data, data_len));
    return 0;
}
