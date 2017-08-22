#pragma once
#include "fwd.h"

#include <util/system/rwlock.h>
#include <util/stream/output.h>
#include <util/generic/hash.h>
#include <util/generic/string.h>

namespace NColorizer {
    class TColors: private TNonCopyable {
    public:
        explicit TColors(FILE* = stderr);
        explicit TColors(bool ontty);

        /*
         * ANSI escape codes helpers
         * All functions returns zero-terminated TStringBuf
         */
        TStringBuf OldColor() const;
        TStringBuf BoldColor() const;
        TStringBuf BlackColor() const;
        TStringBuf BlueColor() const;
        TStringBuf GreenColor() const;
        TStringBuf CyanColor() const;
        TStringBuf RedColor() const;
        TStringBuf PurpleColor() const;
        TStringBuf BrownColor() const;
        TStringBuf LightGrayColor() const;
        TStringBuf DarkGrayColor() const;
        TStringBuf LightBlueColor() const;
        TStringBuf LightGreenColor() const;
        TStringBuf LightCyanColor() const;
        TStringBuf LightRedColor() const;
        TStringBuf LightPurpleColor() const;
        TStringBuf YellowColor() const;
        TStringBuf WhiteColor() const;

        TStringBuf Color(const TStringBuf& name) const;

        inline bool IsTTY() const noexcept {
            return IsTTY_;
        }

        static bool CalcIsTTY(FILE* file);

        inline void SetIsTTY(bool value) noexcept {
            IsTTY_ = value;
        }

        inline void Enable() noexcept {
            SetIsTTY(true);
        }

        inline void Disable() noexcept {
            SetIsTTY(false);
        }

    private:
        void InitNames();
        TStringBuf ColorImpl(int val, int bold) const;

        inline TStringBuf Color(int val, int bold = 0) const {
            return IsTTY() ? ColorImpl(val, bold) : TStringBuf("");
        }

    private:
        bool IsTTY_;
        mutable yhash<int, TString> ColorBufs;
        yhash<TStringBuf, TStringBuf> ByName;
        TRWMutex Mutex;
    };

    //stderr/stdout synced colors
    TColors& StdErr();
    TColors& StdOut();

    // choose TColors if os is Cerr or Cout
    TColors& AutoColors(IOutputStream& os);
}
