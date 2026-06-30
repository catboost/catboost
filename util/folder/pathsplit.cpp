#include "pathsplit.h"

#include <util/stream/output.h>
#include <util/generic/yexception.h>

template <class T>
static inline size_t ToReserve(const T& t) {
    size_t ret = t.size() + 5;

    for (auto it = t.begin(); it != t.end(); ++it) {
        ret += it->size();
    }

    return ret;
}

void TPathSplitTraitsUnix::DoParseFirstPart(const TStringBuf part) {
    if (part == TStringBuf(".")) {
        push_back(TStringBuf("."));

        return;
    }

    if (IsAbsolutePath(part)) {
        IsAbsolute = true;
    }

    DoParsePart(part);
}

void TPathSplitTraitsUnix::DoParsePart(const TStringBuf part0) {
    DoAppendHint(part0.size() / 8);

    TStringBuf next(part0);
    TStringBuf part;

    while (TStringBuf(next).TrySplit('/', part, next)) {
        AppendComponent(part);
    }

    AppendComponent(next);
}

void TPathSplitTraitsWindows::DoParseFirstPart(const TStringBuf part0) {
    TStringBuf part(part0);

    if (part == TStringBuf(".")) {
        push_back(TStringBuf("."));

        return;
    }

    if (IsAbsolutePath(part)) {
        IsAbsolute = true;

        if (part.size() > 1 && part[1] == ':') {
            Drive = part.SubStr(0, 2);
            part = part.SubStr(2);
        }
    }

    DoParsePart(part);
}

void TPathSplitTraitsWindows::DoParsePart(const TStringBuf part0) {
    DoAppendHint(part0.size() / 8);

    size_t pos = 0;
    TStringBuf part(part0);

    while (pos < part.size()) {
        while (pos < part.size() && this->IsPathSep(part[pos])) {
            ++pos;
        }

        const char* begin = part.data() + pos;

        while (pos < part.size() && !this->IsPathSep(part[pos])) {
            ++pos;
        }

        AppendComponent(TStringBuf(begin, part.data() + pos));
    }
}

TString TPathSplitStore::DoReconstruct(const TStringBuf slash) const {
    TString r;

    r.reserve(ToReserve(*this));

    if (IsAbsolute) {
        r.AppendNoAlias(Drive);
        r.AppendNoAlias(slash);
    }

    for (auto i = begin(); i != end(); ++i) {
        if (i != begin()) {
            r.AppendNoAlias(slash);
        }

        r.AppendNoAlias(*i);
    }

    return r;
}

void TPathSplitStore::AppendComponent(const TStringBuf comp) {
    if (!comp || comp == TStringBuf(".")) {
        // ignore
    } else if (comp == TStringBuf("..") && !empty() && back() != TStringBuf("..")) {
        pop_back();
    } else {
        // push back first .. also
        push_back(comp);
    }
}

TStringBuf TPathSplitStore::Extension() const {
    return size() > 0 ? CutExtension(back()) : TStringBuf();
}

template <>
void Out<TPathSplit>(IOutputStream& o, const TPathSplit& ps) {
    o << ps.Reconstruct();
}

TString JoinPaths(const TPathSplit& p1, const TPathSplit& p2) {
    if (p2.IsAbsolute) {
        ythrow yexception() << "can not join " << p1 << " and " << p2;
    }

    return TPathSplit(p1).AppendMany(p2.begin(), p2.end()).Reconstruct();
}

TStringBuf CutExtension(const TStringBuf fileName Y_LIFETIME_BOUND) {
    if (fileName.empty()) {
        return fileName;
    }

    TStringBuf name;
    TStringBuf extension;
    fileName.RSplit('.', name, extension);
    if (name.empty()) {
        // dot at a start or not found
        return name;
    } else {
        return extension;
    }
}
