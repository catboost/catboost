#pragma once

#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/stream/output.h>

class TCppSaxParser: public IOutputStream {
public:
    struct TText {
        TText();
        TText(ui64 offset);
        TText(const TString& data, ui64 offset);
        ~TText();

        void Reset() noexcept;

        TString Data;
        ui64 Offset;
    };

    class TWorker {
    public:
        typedef TCppSaxParser::TText TText;

        TWorker() noexcept;
        virtual ~TWorker();

        virtual void DoEnd() = 0;
        virtual void DoStart() = 0;
        virtual void DoString(const TText& text) = 0;
        virtual void DoCharacter(const TText& text) = 0;
        virtual void DoCode(const TText& text) = 0;
        virtual void DoOneLineComment(const TText& text) = 0;
        virtual void DoMultiLineComment(const TText& text) = 0;
        virtual void DoPreprocessor(const TText& text) = 0;
    };

    TCppSaxParser(TWorker* worker);
    ~TCppSaxParser() override;

private:
    void DoWrite(const void* data, size_t len) override;
    void DoFinish() override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

class TCppSimpleSax: public TCppSaxParser::TWorker {
public:
    TCppSimpleSax() noexcept;
    ~TCppSimpleSax() override;

    void DoEnd() override = 0;
    void DoStart() override = 0;
    void DoString(const TText& text) override = 0;
    void DoCharacter(const TText& text) override = 0;
    virtual void DoWhiteSpace(const TText& text) = 0;
    virtual void DoIdentifier(const TText& text) = 0;
    virtual void DoSyntax(const TText& text) = 0;
    void DoOneLineComment(const TText& text) override = 0;
    void DoMultiLineComment(const TText& text) override = 0;
    void DoPreprocessor(const TText& text) override = 0;

private:
    void DoCode(const TText& text) override;
};

class TCppFullSax: public TCppSimpleSax {
public:
    TCppFullSax();
    ~TCppFullSax() override;

    void DoEnd() override;
    void DoStart() override;
    void DoString(const TText& text) override;
    void DoCharacter(const TText& text) override;
    void DoWhiteSpace(const TText& text) override;
    virtual void DoKeyword(const TText& text);
    virtual void DoName(const TText& text);
    virtual void DoOctNumber(const TText& text);
    virtual void DoHexNumber(const TText& text);
    virtual void DoDecNumber(const TText& text);
    virtual void DoFloatNumber(const TText& text);
    void DoSyntax(const TText& text) override;
    void DoOneLineComment(const TText& text) override;
    void DoMultiLineComment(const TText& text) override;
    void DoPreprocessor(const TText& text) override;

    void AddKeyword(const TString& keyword);

private:
    void DoIdentifier(const TText& text) override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};
