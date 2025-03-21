#include "registry.h"

#include <library/cpp/containers/absl_flat_hash/flat_hash_map.h>
#include <library/cpp/blockcodecs/core/codecs.h>

#include <util/system/yassert.h>
#include <util/generic/hash.h>
#include <util/generic/deque.h>
#include <util/generic/singleton.h>
#include <util/system/env.h>

using namespace NResource;
using namespace NBlockCodecs;

namespace {
    inline const ICodec* GetCodec() noexcept {
        static const ICodec* ret = Codec("zstd08_5");

        return ret;
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

    struct TStore final: public IStore, public absl::flat_hash_map<ui64, TDescriptor*> {
        static inline ui64 ToK(TStringBuf k) {
            return NHashPrivate::ComputeStringHash(k.data(), k.size());
        }

        void Store(const TStringBuf key, const TStringBuf data) override {
            auto kk = ToK(key);

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

            Y_ABORT_UNLESS(size() == Count(), "size mismatch");
        }

        bool Has(const TStringBuf key) const override {
            return contains(ToK(key));
        }

        bool FindExact(const TStringBuf key, TString* out) const override {
            if (auto res = find(ToK(key)); res != end()) {
                // temporary
                // https://st.yandex-team.ru/DEVTOOLS-3985
                try {
                    *out = Decompress(res->second->second);
                } catch (const yexception& e) {
                    if (GetEnv("RESOURCE_DECOMPRESS_DIAG")) {
                        Cerr << "Can't decompress resource " << key << Endl << e.what() << Endl;
                    }
                    throw e;
                }

                return true;
            }

            return false;
        }

        void FindMatch(const TStringBuf subkey, IMatch& cb) const override {
            for (const auto& it : D_) {
                if (it.first.StartsWith(subkey)) {
                    // temporary
                    // https://st.yandex-team.ru/DEVTOOLS-3985
                    try {
                        const TResource res = {
                            it.first, Decompress(it.second)};
                        cb.OnMatch(res);
                    } catch (const yexception& e) {
                        if (GetEnv("RESOURCE_DECOMPRESS_DIAG")) {
                            Cerr << "Can't decompress resource " << it.first << Endl << e.what() << Endl;
                        }
                        throw e;
                    }
                }
            }
        }

        size_t Count() const noexcept override {
            return D_.size();
        }

        TStringBuf KeyByIndex(size_t idx) const override {
            return D_.at(idx).first;
        }

        typedef TDeque<TDescriptor> TDescriptors;
        TDescriptors D_;
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
