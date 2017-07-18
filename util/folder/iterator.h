#pragma once

#include "fts.h"

#include <util/system/error.h>
#include <util/generic/ptr.h>
#include <util/generic/iterator.h>
#include <util/generic/yexception.h>

/// Note this magic API traverses directory hierarchy

class TDirIterator: public TStlIterator<TDirIterator> {
    struct TFtsDestroy {
        static inline void Destroy(FTS* f) noexcept {
            yfts_close(f);
        }
    };

public:
    class TError: public TSystemError {
    public:
        inline TError(int err)
            : TSystemError(err)
        {
        }
    };

    using TCompare = int (*)(const FTSENT**, const FTSENT**);

    struct TOptions {
        inline TOptions() {
            Init(FTS_PHYSICAL);
        }

        inline TOptions(int opts) {
            Init(opts);
        }

        inline TOptions& SetMaxLevel(size_t level) noexcept {
            MaxLevel = level;

            return *this;
        }

        inline TOptions& SetSortFunctor(TCompare cmp) noexcept {
            Cmp = cmp;

            return *this;
        }

        TOptions& SetSortByName() noexcept;

        int FtsOptions;
        size_t MaxLevel;
        TCompare Cmp;

    private:
        inline void Init(int opts) noexcept {
            FtsOptions = opts | FTS_NOCHDIR;
            MaxLevel = Max<size_t>();
            Cmp = nullptr;
        }
    };

    using TRetVal = FTSENT*;

    inline TDirIterator(const TString& path, const TOptions& options = TOptions())
        : Options_(options)
        , Path_(path)
    {
        Trees_[0] = Path_.begin();
        Trees_[1] = nullptr;

        ClearLastSystemError();
        FileTree_.Reset(yfts_open(Trees_, Options_.FtsOptions, Options_.Cmp));

        const int err = LastSystemError();

        if (err) {
            ythrow TError(err) << "can not open '" << Path_ << "'";
        }
    }

    inline FTSENT* Next() {
        FTSENT* ret = yfts_read(FileTree_.Get());

        if (ret) {
            if ((size_t)(ret->fts_level + 1) > Options_.MaxLevel) {
                yfts_set(FileTree_.Get(), ret, FTS_SKIP);
            }
        } else {
            const int err = LastSystemError();

            if (err) {
                ythrow TError(err) << "error while iterating " << Path_;
            }
        }

        return ret;
    }

private:
    TOptions Options_;
    TString Path_;
    char* Trees_[2];
    THolder<FTS, TFtsDestroy> FileTree_;
};
