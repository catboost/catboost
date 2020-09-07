%{
#include <util/generic/ptr.h>
%}

template <class T>
class TIntrusivePtr {
    T *pointee;
public:
    T *operator->() const noexcept {
        return pointee;
    }
};