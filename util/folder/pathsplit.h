#pragma once

#include <util/generic/vector.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/string/ascii.h>

// do not own any data
struct TPathSplitStore: public TVector<TStringBuf> {
    TStringBuf Drive;
    bool IsAbsolute = false;

    void AppendComponent(const TStringBuf comp);
    TStringBuf Extension() const;

protected:
    TString DoReconstruct(const TStringBuf slash) const;

    inline void DoAppendHint(size_t hint) {
        reserve(size() + hint);
    }
};

struct TPathSplitTraitsUnix: public TPathSplitStore {
    static constexpr char MainPathSep = '/';

    inline TString Reconstruct() const {
        return DoReconstruct(TStringBuf("/"));
    }

    static constexpr bool IsPathSep(const char c) noexcept {
        return c == '/';
    }

    static inline bool IsAbsolutePath(const TStringBuf path) noexcept {
        return path && IsPathSep(path[0]);
    }

    void DoParseFirstPart(const TStringBuf part);
    void DoParsePart(const TStringBuf part);
};

struct TPathSplitTraitsWindows: public TPathSplitStore {
    static constexpr char MainPathSep = '\\';

    inline TString Reconstruct() const {
        return DoReconstruct(TStringBuf("\\"));
    }

    static constexpr bool IsPathSep(char c) noexcept {
        return c == '/' || c == '\\';
    }

    static inline bool IsAbsolutePath(const TStringBuf path) noexcept {
        return path && (IsPathSep(path[0]) || (path.size() > 1 && path[1] == ':' && IsAsciiAlpha(path[0]) && (path.size() == 2 || IsPathSep(path[2]))));
    }

    void DoParseFirstPart(const TStringBuf part);
    void DoParsePart(const TStringBuf part);
};

#if defined(_unix_)
using TPathSplitTraitsLocal = TPathSplitTraitsUnix;
#else
using TPathSplitTraitsLocal = TPathSplitTraitsWindows;
#endif

template <class TTraits>
class TPathSplitBase: public TTraits {
public:
    inline TPathSplitBase() = default;

    inline TPathSplitBase(const TStringBuf part) {
        this->ParseFirstPart(part);
    }

    inline TPathSplitBase& AppendHint(size_t hint) {
        this->DoAppendHint(hint);

        return *this;
    }

    inline TPathSplitBase& ParseFirstPart(const TStringBuf part) {
        this->DoParseFirstPart(part);

        return *this;
    }

    inline TPathSplitBase& ParsePart(const TStringBuf part) {
        this->DoParsePart(part);

        return *this;
    }

    template <class It>
    inline TPathSplitBase& AppendMany(It b, It e) {
        this->AppendHint(e - b);

        while (b != e) {
            this->AppendComponent(*b++);
        }

        return *this;
    }
};

using TPathSplit = TPathSplitBase<TPathSplitTraitsLocal>;
using TPathSplitUnix = TPathSplitBase<TPathSplitTraitsUnix>;
using TPathSplitWindows = TPathSplitBase<TPathSplitTraitsWindows>;

TString JoinPaths(const TPathSplit& p1, const TPathSplit& p2);

TStringBuf CutExtension(const TStringBuf fileName Y_LIFETIME_BOUND);
