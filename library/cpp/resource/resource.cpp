#include "resource.h"
#include "resource.h"
#include "registry.h"

#include <util/generic/yexception.h>
#include <util/generic/xrange.h>

using namespace NResource;

bool NResource::FindExact(const TStringBuf key, TString* out) {
    return CommonStore()->FindExact(key, out);
}

void NResource::FindMatch(const TStringBuf subkey, TResources* out) {
    struct TMatch: public IMatch {
        inline TMatch(TResources* r)
            : R(r)
        {
        }

        void OnMatch(const TResource& res) override {
            R->push_back(res);
        }

        TResources* R;
    };

    TMatch m(out);

    CommonStore()->FindMatch(subkey, m);
}

bool NResource::Has(const TStringBuf key) {
    return CommonStore()->Has(key);
}

TString NResource::Find(const TStringBuf key) {
    TString ret;

    if (FindExact(key, &ret)) {
        return ret;
    }

    ythrow yexception() << "can not find resource with path " << key;
}

size_t NResource::Count() noexcept {
    return CommonStore()->Count();
}

TStringBuf NResource::KeyByIndex(size_t idx) {
    return CommonStore()->KeyByIndex(idx);
}

TVector<TStringBuf> NResource::ListAllKeys() {
    TVector<TStringBuf> res(Reserve(NResource::Count()));
    for (auto i : xrange(NResource::Count())) {
        res.push_back(NResource::KeyByIndex(i));
    }
    return res;
}
