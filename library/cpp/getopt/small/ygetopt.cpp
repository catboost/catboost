#include "opt.h"
#include "ygetopt.h"

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>

class TGetOpt::TImpl: public TSimpleRefCount<TImpl> {
public:
    inline TImpl(int argc, const char* const* argv, const TString& fmt)
        : args(argv, argv + argc)
        , format(fmt)
    {
        if (argc == 0) {
            ythrow yexception() << "zero argc";
        }
    }

    inline ~TImpl() = default;

    TVector<TString> args;
    const TString format;
};

class TGetOpt::TIterator::TIterImpl: public TSimpleRefCount<TIterImpl> {
public:
    inline TIterImpl(const TGetOpt* parent)
        : Args_(parent->Impl_->args)
        , ArgsPtrs_(new char*[Args_.size() + 1])
        , Format_(parent->Impl_->format)
        , OptLet_(0)
        , Arg_(nullptr)
    {
        for (size_t i = 0; i < Args_.size(); ++i) {
            ArgsPtrs_.Get()[i] = Args_[i].begin();
        }

        ArgsPtrs_.Get()[Args_.size()] = nullptr;
        Opt_.Reset(new Opt((int)Args_.size(), ArgsPtrs_.Get(), Format_.data()));
    }

    inline ~TIterImpl() = default;

    inline void Next() {
        OptLet_ = Opt_->Get();
        Arg_ = Opt_->Arg;
    }

    inline char Key() const noexcept {
        return (char)OptLet_;
    }

    inline const char* Arg() const noexcept {
        return Arg_;
    }

    inline bool AtEnd() const noexcept {
        return OptLet_ == EOF;
    }

private:
    TVector<TString> Args_;
    TArrayHolder<char*> ArgsPtrs_;
    const TString Format_;
    THolder<Opt> Opt_;
    int OptLet_;
    const char* Arg_;
};

TGetOpt::TIterator::TIterator() noexcept
    : Impl_(nullptr)
{
}

TGetOpt::TIterator::TIterator(const TGetOpt* parent)
    : Impl_(new TIterImpl(parent))
{
    Next();
}

void TGetOpt::TIterator::Next() {
    Impl_->Next();
}

char TGetOpt::TIterator::Key() const noexcept {
    return Impl_->Key();
}

bool TGetOpt::TIterator::AtEnd() const noexcept {
    if (Impl_.Get()) {
        return Impl_->AtEnd();
    }

    return true;
}

const char* TGetOpt::TIterator::Arg() const noexcept {
    if (Impl_.Get()) {
        return Impl_->Arg();
    }

    return nullptr;
}

TGetOpt::TGetOpt(int argc, const char* const* argv, const TString& format)
    : Impl_(new TImpl(argc, argv, format))
{
}
