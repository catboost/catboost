#pragma once

#include "concepts/size_fitter.h"

#include <util/system/yassert.h>
#include <util/generic/bitops.h>

namespace NFlatHash {

class TAndSizeFitter {
public:
    size_t EvalIndex(size_t hs, size_t sz) const noexcept {
        Y_ASSERT(Mask_ == sz - 1);
        return (hs & Mask_);
    }

    size_t EvalSize(size_t sz) const noexcept {
        return FastClp2(sz);
    }

    void Update(size_t sz) noexcept {
        Y_ASSERT((sz & (sz - 1)) == 0);
        Mask_ = sz - 1;
    }

private:
    size_t Mask_ = 0;
};

static_assert(NConcepts::SizeFitterV<TAndSizeFitter>);

class TModSizeFitter {
public:
    constexpr size_t EvalIndex(size_t hs, size_t sz) const noexcept {
        return hs % sz;
    }

    constexpr size_t EvalSize(size_t sz) const noexcept {
        return sz;
    }

    constexpr void Update(size_t) noexcept {}
};

static_assert(NConcepts::SizeFitterV<TModSizeFitter>);

}  // NFlatHash
