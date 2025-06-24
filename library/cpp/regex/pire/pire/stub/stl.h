#ifndef PIRE_COMPAT_H_INCLUDED
#define PIRE_COMPAT_H_INCLUDED

#include <bitset>
#include <algorithm>
#include <iterator>
#include <functional>
#include <utility>
#include <memory>

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/deque.h>
#include <util/generic/list.h>
#include <util/generic/map.h>
#include <util/generic/set.h>
#include <util/generic/hash.h>
#include <util/generic/hash_set.h>
#include <util/generic/ptr.h>
#include <util/generic/yexception.h>
#include <util/generic/utility.h>
#include <util/generic/algorithm.h>
#include <util/stream/input.h>
#include <util/stream/output.h>
#include <util/string/reverse.h>
#include <util/string/vector.h>

namespace Pire {
    using ystring = TString;
    template<size_t N> using ybitset = std::bitset<N>;
    template<typename T1, typename T2> using ypair = std::pair<T1, T2>;
    template<typename T> using yauto_ptr = std::auto_ptr<T>;
    template<typename Arg1, typename Arg2, typename Result> using ybinary_function = std::binary_function<Arg1, Arg2, Result>;

    template<typename T1, typename T2>
    inline ypair<T1, T2> ymake_pair(T1 v1, T2 v2) {
        return  std::make_pair(v1, v2);
    }

    template<typename T>
    inline T ymax(T v1, T v2) {
        return std::max(v1, v2);
    }

    template<typename T>
    inline T ymin(T v1, T v2) {
        return std::min(v1, v2);
    }

    template<class Iter, class T>
    void Fill(Iter begin, Iter end, T t) { std::fill(begin, end, t); }

    class Error: public yexception {
    public:
        Error(const char* msg)    { *this << msg; }
        Error(const ystring& msg) { *this << msg; }
    };

    typedef IOutputStream yostream;
    typedef IInputStream yistream;

    template<class Iter>
    ystring Join(Iter begin, Iter end, const ystring& separator) { return JoinStrings(begin, end, separator); }
}

#endif
