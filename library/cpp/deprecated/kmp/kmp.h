#pragma once

#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>

template <typename T>
void ComputePrefixFunction(const T* begin, const T* end, ssize_t** result) {
    Y_ENSURE(begin != end, TStringBuf("empty pattern"));
    ssize_t len = end - begin;
    TArrayHolder<ssize_t> resultHolder(new ssize_t[len + 1]);
    ssize_t i = 0;
    ssize_t j = -1;
    resultHolder[0] = -1;
    while (i < len) {
        while ((j >= 0) && (begin[j] != begin[i]))
            j = resultHolder[j];
        ++i;
        ++j;
        Y_ASSERT(i >= 0);
        Y_ASSERT(j >= 0);
        Y_ASSERT(j < len);
        if ((i < len) && (begin[i] == begin[j]))
            resultHolder[i] = resultHolder[j];
        else
            resultHolder[i] = j;
    }
    *result = resultHolder.Release();
}

class TKMPMatcher {
private:
    TArrayHolder<ssize_t> PrefixFunction;
    TString Pattern;

    void ComputePrefixFunction();

public:
    TKMPMatcher(const char* patternBegin, const char* patternEnd);
    TKMPMatcher(const TString& pattern);

    bool SubStr(const char* begin, const char* end, const char*& result) const {
        Y_ASSERT(begin <= end);
        ssize_t m = Pattern.size();
        ssize_t n = end - begin;
        ssize_t i, j;
        for (i = 0, j = 0; (i < n) && (j < m); ++i, ++j) {
            while ((j >= 0) && (Pattern[j] != begin[i]))
                j = PrefixFunction[j];
        }
        if (j == m) {
            result = begin + i - m;
            return true;
        } else {
            return false;
        }
    }
};

template <typename T>
class TKMPStreamMatcher {
public:
    class ICallback {
    public:
        virtual void OnMatch(const T* begin, const T* end) = 0;
        virtual ~ICallback() = default;
    };

private:
    ICallback* Callback;
    TArrayHolder<ssize_t> PrefixFunction;
    using TTVector = TVector<T>;
    TTVector Pattern;
    ssize_t State;
    TTVector Candidate;

public:
    TKMPStreamMatcher(const T* patternBegin, const T* patternEnd, ICallback* callback)
        : Callback(callback)
        , Pattern(patternBegin, patternEnd)
        , State(0)
        , Candidate(Pattern.size())
    {
        ssize_t* pf;
        ComputePrefixFunction(patternBegin, patternEnd, &pf);
        PrefixFunction.Reset(pf);
    }

    void Push(const T& symbol) {
        while ((State >= 0) && (Pattern[State] != symbol)) {
            Y_ASSERT(State <= (ssize_t) Pattern.size());
            State = PrefixFunction[State];
            Y_ASSERT(State <= (ssize_t) Pattern.size());
        }
        if (State >= 0)
            Candidate[State] = symbol;
        ++State;
        if (State == (ssize_t) Pattern.size()) {
            Callback->OnMatch(Candidate.begin(), Candidate.end());
            State = 0;
        }
    }

    void Clear() {
        State = 0;
    }
};
