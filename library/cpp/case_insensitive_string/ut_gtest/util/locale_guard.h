#include <util/generic/string.h>
#include <util/system/yassert.h>

#include <cerrno>
#include <clocale>
#include <cstring>

struct [[nodiscard]] TLocaleGuard {
    TLocaleGuard(const char* loc) {
        PrevLoc_ = std::setlocale(LC_ALL, nullptr);
        const char* res = std::setlocale(LC_ALL, loc);
        if (!res) {
            Error_ = std::strerror(errno);
        }
    }

    ~TLocaleGuard() {
        if (!Error_) {
            Y_ABORT_UNLESS(std::setlocale(LC_ALL, PrevLoc_.c_str()));
        }
    }

    const TString& Error() const noexcept {
        return Error_;
    }

private:
    // "POSIX also specifies that the returned pointer, not just the contents of the pointed-to string, may be invalidated by subsequent calls to setlocale".
    TString PrevLoc_;
    TString Error_;
};
