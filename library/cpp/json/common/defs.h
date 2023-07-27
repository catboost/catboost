#pragma once

#include <util/generic/strbuf.h>
#include <util/generic/yexception.h>

namespace NJson {
    class TJsonException: public yexception {
    };

    class TJsonCallbacks {
    public:
        explicit TJsonCallbacks(bool throwException = false)
            : ThrowException(throwException)
        {
        }

        virtual ~TJsonCallbacks();

        virtual bool OnNull();
        virtual bool OnBoolean(bool);
        virtual bool OnInteger(long long);
        virtual bool OnUInteger(unsigned long long);
        virtual bool OnDouble(double);
        virtual bool OnString(const TStringBuf&);
        virtual bool OnOpenMap();
        virtual bool OnMapKey(const TStringBuf&);
        virtual bool OnCloseMap();
        virtual bool OnOpenArray();
        virtual bool OnCloseArray();
        virtual bool OnStringNoCopy(const TStringBuf& s);
        virtual bool OnMapKeyNoCopy(const TStringBuf& s);
        virtual bool OnEnd();
        virtual void OnError(size_t off, TStringBuf reason);

        bool GetHaveErrors() const {
            return HaveErrors;
        }
    protected:
        bool ThrowException;
        bool HaveErrors = false;
    };
}
