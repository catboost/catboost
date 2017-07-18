#pragma once

#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/system/compiler.h>

namespace NSubst {
    namespace NPrivate {
        // a bit of template magic (to be fast and unreadable)
        template <class TStroka, class TTo, bool Main>
        static Y_FORCE_INLINE void MoveBlock(typename TStroka::value_type* ptr, size_t& srcPos, size_t& dstPos, const size_t off, const TTo& to, const size_t toSize) {
            const size_t unchangedSize = off - srcPos;
            if (dstPos < srcPos) {
                for (size_t i = 0; i < unchangedSize; ++i) {
                    ptr[dstPos++] = ptr[srcPos++];
                }
            } else {
                dstPos += unchangedSize;
                srcPos += unchangedSize;
            }

            if (Main) {
                for (size_t i = 0; i < toSize; ++i) {
                    ptr[dstPos++] = to[i];
                }
            }
        }
    }; // NSubst::NPrivate
};     // NSubst

/**
 * Replaces all occurences of substring @c from in string @c s to string @c to.
 * Uses two separate implementations (inplace for shrink and append for grow case)
 * See IGNIETFERRO-394
 **/
template <class TStroka, class TFrom, class TTo>
inline size_t SubstGlobalImpl(TStroka& s, const TFrom& from, const TTo& to, size_t fromPos = 0) {
    if (!from) {
        return 0;
    }
    const size_t fromSize = from.size();
    const size_t toSize = to.size();
    size_t replacementsCount = 0;
    size_t off = fromPos;
    size_t srcPos = 0;

    if (toSize > fromSize) {
        // string will grow: append to another string
        TStroka result;
        for (; (off = s.find(from, off)) != TStroka::npos; off += fromSize) {
            if (!replacementsCount) {
                // first replacement occured, we can prepare result string
                result.reserve(s.size() + s.size() / 3);
            }
            result.append(s.begin() + srcPos, s.begin() + off);
            result.append(~to, +to);
            srcPos = off + fromSize;
            ++replacementsCount;
        }
        if (replacementsCount) {
            // append tail
            result.append(s.begin() + srcPos, s.end());
            s = std::move(result);
        }
        return replacementsCount;
    }

    // string will not grow: use inplace algo
    size_t dstPos = 0;
    typename TStroka::value_type* ptr = s.begin();
    for (; (off = s.find(from, off)) != TStroka::npos; off += fromSize) {
        Y_ASSERT(dstPos <= srcPos);
        NSubst::NPrivate::MoveBlock<TStroka, TTo, true>(ptr, srcPos, dstPos, off, to, toSize);
        srcPos = off + fromSize;
        ++replacementsCount;
    }

    if (replacementsCount) {
        // append tail
        NSubst::NPrivate::MoveBlock<TStroka, TTo, false>(ptr, srcPos, dstPos, s.size(), to, toSize);
        s.resize(dstPos);
    }
    return replacementsCount;
}

template <class TStroka, class TFrom, class TTo>
inline size_t SubstGlobal(TStroka& s, const TFrom& from, const TTo& to, size_t fromPos = 0) {
    return SubstGlobalImpl(s, typename TToStringBuf<TStroka>::TType(from), typename TToStringBuf<TStroka>::TType(to), fromPos);
}

inline size_t SubstGlobal(TString& s, const TString& from, const TString& to, size_t fromPos = 0) {
    return SubstGlobal<TString>(s, from, to, fromPos);
}

/// Replaces all occurences of the 'from' symbol in a string to the 'to' symbol.
template <class TStroka>
inline size_t SubstCharGlobalImpl(TStroka& s, typename TStroka::char_type from, typename TStroka::char_type to, size_t fromPos = 0) {
    if (fromPos >= s.size()) {
        return 0;
    }

    size_t result = 0;
    fromPos = s.find(from, fromPos);

    // s.begin() might cause memory copying, so call it only if needed
    if (fromPos != TStroka::npos) {
        auto* it = s.begin() + fromPos;
        *it = to;
        ++result;
        // at this point string is copied and it's safe to use constant s.end() to iterate
        const auto* const sEnd = s.end();
        // unrolled loop goes first because it is more likely that `it` will be properly aligned
        for (const auto* const end = sEnd - (sEnd - it) % 4; it < end;) {
            if (*it == from) {
                *it = to;
                ++result;
            }
            ++it;
            if (*it == from) {
                *it = to;
                ++result;
            }
            ++it;
            if (*it == from) {
                *it = to;
                ++result;
            }
            ++it;
            if (*it == from) {
                *it = to;
                ++result;
            }
            ++it;
        }
        for (; it < sEnd; ++it) {
            if (*it == from) {
                *it = to;
                ++result;
            }
        }
    }

    return result;
}

inline size_t SubstGlobal(TString& s, char from, char to, size_t fromPos = 0) {
    return SubstCharGlobalImpl<TString>(s, from, to, fromPos);
}

inline size_t SubstGlobal(TUtf16String& s, TChar from, TChar to, size_t fromPos = 0) {
    return SubstCharGlobalImpl<TUtf16String>(s, from, to, fromPos);
}
