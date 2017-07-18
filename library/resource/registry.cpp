#include "registry.h"

#include <library/blockcodecs/codecs.h>

#include <util/system/yassert.h>
#include <util/generic/hash.h>
#include <util/generic/deque.h>
#include <util/generic/singleton.h>

using namespace NResource;
using namespace NBlockCodecs;

namespace {
    static inline const ICodec* GetCodec() noexcept {
        static const ICodec* ret = Codec("zstd08_5");

        return ret;
    }

    typedef std::pair<TStringBuf, TStringBuf> TDescriptor;

    struct TStore: public IStore, public yhash<TStringBuf, TDescriptor*> {
        void Store(const TStringBuf& key, const TStringBuf& data) override {
            if (has(key)) {
                if ((*this)[key]->second != data) {
                    Y_VERIFY(false, "Multiple definition for key '%s'", ~key);
                }
            } else {
                D_.push_back(TDescriptor(key, data));
                (*this)[key] = &D_.back();
            }

            Y_VERIFY(size() == Count(), "shit happen");
        }

        bool FindExact(const TStringBuf& key, TString* out) const override {
            if (TDescriptor* const* res = FindPtr(key)) {
                *out = Decompress((*res)->second);

                return true;
            }

            return false;
        }

        void FindMatch(const TStringBuf& subkey, IMatch& cb) const override {
            for (const auto& it : *this) {
                if (it.first.StartsWith(subkey)) {
                    const TResource res = {
                        it.first, Decompress(it.second->second)};

                    cb.OnMatch(res);
                }
            }
        }

        size_t Count() const noexcept override {
            return D_.size();
        }

        TStringBuf KeyByIndex(size_t idx) const override {
            return D_.at(idx).first;
        }

        typedef ydeque<TDescriptor> TDescriptors;
        TDescriptors D_;
    };
}

TString NResource::Compress(const TStringBuf& data) {
    return GetCodec()->Encode(data);
}

TString NResource::Decompress(const TStringBuf& data) {
    return GetCodec()->Decode(data);
}

IStore* NResource::CommonStore() {
    return Singleton<TStore>();
}
