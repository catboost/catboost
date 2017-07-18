#include "split.h"

template <class TValue>
inline size_t Split(const char* ptr, const char* delim, yvector<TValue>& values) {
    values.erase(values.begin(), values.end());
    while (ptr && *ptr) {
        ptr += strspn(ptr, delim);
        if (ptr && *ptr) {
            size_t epos = strcspn(ptr, delim);
            assert(epos);
            values.push_back(TValue(ptr, epos));
            ptr += epos;
        }
    }
    return values.size();
}

size_t Split(const char* ptr, const char* delim, yvector<TString>& values) {
    return Split<TString>(ptr, delim, values);
}

size_t Split(const char* ptr, const char* delim, yvector<TStringBuf>& values) {
    return Split<TStringBuf>(ptr, delim, values);
}

size_t Split(const TString& in, const TString& delim, yvector<TString>& res) {
    return Split(~in, ~delim, res);
}
