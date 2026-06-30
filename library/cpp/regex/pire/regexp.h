#pragma once

#include "pire.h"

#include <library/cpp/charset/doccodes.h>
#include <library/cpp/charset/recyr.hh>
#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>

namespace NRegExp {
    struct TMatcher;

    struct TFsmBase {
        struct TOptions {
            inline TOptions& SetCaseInsensitive(bool v) noexcept {
                CaseInsensitive = v;
                return *this;
            }

            inline TOptions& SetSurround(bool v) noexcept {
                Surround = v;
                return *this;
            }

            inline TOptions& SetCapture(size_t pos) noexcept {
                CapturePos = pos;
                return *this;
            }

            inline TOptions& SetCharset(ECharset charset) noexcept {
                Charset = charset;
                return *this;
            }

            inline TOptions& SetAndNotSupport(bool andNotSupport) noexcept {
                AndNotSupport = andNotSupport;
                return *this;
            }

            bool CaseInsensitive = false;
            bool Surround = false;
            TMaybe<size_t> CapturePos;
            ECharset Charset = CODES_UNKNOWN;
            bool AndNotSupport = false;
        };

        static inline NPire::TFsm Parse(const TStringBuf& regexp,
                                        const TOptions& opts, const bool needDetermine = true) {
            NPire::TLexer lexer;
            if (opts.Charset == CODES_UNKNOWN) {
                lexer.Assign(regexp.data(), regexp.data() + regexp.size());
            } else {
                TVector<wchar32> ucs4(regexp.size() + 1);
                size_t inRead = 0;
                size_t outWritten = 0;
                int recodeRes = RecodeToUnicode(opts.Charset, regexp.data(), ucs4.data(),
                                                regexp.size(), regexp.size(), inRead, outWritten);
                Y_ASSERT(recodeRes == RECODE_OK);
                Y_ASSERT(outWritten < ucs4.size());
                ucs4[outWritten] = 0;

                lexer.Assign(ucs4.begin(),
                             ucs4.begin() + std::char_traits<wchar32>::length(ucs4.data()));
            }

            if (opts.CaseInsensitive) {
                lexer.AddFeature(NPire::NFeatures::CaseInsensitive());
            }

            if (opts.CapturePos) {
                lexer.AddFeature(NPire::NFeatures::Capture(*opts.CapturePos));
            }

            if (opts.AndNotSupport) {
                lexer.AddFeature(NPire::NFeatures::AndNotSupport());
            }

            switch (opts.Charset) {
                case CODES_UNKNOWN:
                    break;
                case CODES_UTF8:
                    lexer.SetEncoding(NPire::NEncodings::Utf8());
                    break;
                case CODES_KOI8:
                    lexer.SetEncoding(NPire::NEncodings::Koi8r());
                    break;
                default:
                    lexer.SetEncoding(NPire::NEncodings::Get(opts.Charset));
                    break;
            }

            NPire::TFsm ret = lexer.Parse();

            if (opts.Surround) {
                ret.Surround();
            }

            if (needDetermine) {
                ret.Determine();
            }

            return ret;
        }
    };

    template <class TScannerType>
    class TFsmParser: public TFsmBase {
    public:
        typedef TScannerType TScanner;

    public:
        inline explicit TFsmParser(const TStringBuf& regexp,
                                   const TOptions& opts = TOptions(), bool needDetermine = true)
            : Scanner(Parse(regexp, opts, needDetermine).template Compile<TScanner>())
        {
        }

        inline const TScanner& GetScanner() const noexcept {
            return Scanner;
        }

        static inline TFsmParser False() {
            return TFsmParser(NPire::TFsm::MakeFalse().Compile<TScanner>());
        }

        inline explicit TFsmParser(const TScanner& compiled)
            : Scanner(compiled)
        {
            if (Scanner.Empty())
                ythrow yexception() << "Can't create fsm with empty scanner";
        }

    private:
        TScanner Scanner;
    };

    class TFsm: public TFsmParser<NPire::TNonrelocScanner> {
    public:
        inline explicit TFsm(const TStringBuf& regexp,
                             const TOptions& opts = TOptions())
            : TFsmParser<TScanner>(regexp, opts)
        {
        }

        inline TFsm(const TFsmParser<TScanner>& fsm)
            : TFsmParser<TScanner>(fsm)
        {
        }

        static inline TFsm Glue(const TFsm& l, const TFsm& r) {
            return TFsm(TScanner::Glue(l.GetScanner(), r.GetScanner()));
        }

        inline explicit TFsm(const TScanner& compiled)
            : TFsmParser<TScanner>(compiled)
        {
        }
    };

    static inline TFsm operator|(const TFsm& l, const TFsm& r) {
        return TFsm::Glue(l, r);
    }

    struct TCapturingFsm : TFsmParser<NPire::TCapturingScanner> {
        inline explicit TCapturingFsm(const TStringBuf& regexp,
                                      TOptions opts = TOptions())
            : TFsmParser<TScanner>(regexp,
                                   opts.SetSurround(true).CapturePos ? opts : opts.SetCapture(1)) {
        }

        inline TCapturingFsm(const TFsmParser<TScanner>& fsm)
            : TFsmParser<TScanner>(fsm)
        {
        }
    };

    struct TSlowCapturingFsm : TFsmParser<NPire::TSlowCapturingScanner> {
        inline explicit TSlowCapturingFsm(const TStringBuf& regexp,
                                          TOptions opts = TOptions())
                : TFsmParser<TScanner>(regexp,
                                       opts.SetSurround(true).CapturePos ? opts : opts.SetCapture(1), false) {
        }

        inline TSlowCapturingFsm(const TFsmParser<TScanner>& fsm)
                : TFsmParser<TScanner>(fsm)
        {
        }
    };

    template <class TFsm>
    class TMatcherBase {
    public:
        typedef typename TFsm::TScanner::State TState;

    public:
        inline explicit TMatcherBase(const TFsm& fsm)
            : Fsm(fsm)
        {
            Fsm.GetScanner().Initialize(State);
        }

        inline bool Final() const noexcept {
            return GetScanner().Final(GetState());
        }

    protected:
        inline void Run(const char* data, size_t len, bool addBegin, bool addEnd) noexcept {
            if (addBegin) {
                NPire::Step(GetScanner(), State, NPire::BeginMark);
            }
            NPire::Run(GetScanner(), State, data, data + len);
            if (addEnd) {
                NPire::Step(GetScanner(), State, NPire::EndMark);
            }
        }

        inline const typename TFsm::TScanner& GetScanner() const noexcept {
            return Fsm.GetScanner();
        }

        inline const TState& GetState() const noexcept {
            return State;
        }

    private:
        const TFsm& Fsm;
        TState State;
    };

    struct TMatcher : TMatcherBase<TFsm> {
        inline explicit TMatcher(const TFsm& fsm)
            : TMatcherBase<TFsm>(fsm)
        {
        }

        inline TMatcher& Match(const char* data, size_t len, bool addBegin = false, bool addEnd = false) noexcept {
            Run(data, len, addBegin, addEnd);
            return *this;
        }

        inline TMatcher& Match(const TStringBuf& s, bool addBegin = false, bool addEnd = false) noexcept {
            return Match(s.data(), s.size(), addBegin, addEnd);
        }

        inline const char* Find(const char* b, const char* e) noexcept {
            return NPire::ShortestPrefix(GetScanner(), b, e);
        }

        typedef std::pair<const size_t*, const size_t*> TMatchedRegexps;

        inline TMatchedRegexps MatchedRegexps() const noexcept {
            return GetScanner().AcceptedRegexps(GetState());
        }
    };

    class TSearcher: public TMatcherBase<TCapturingFsm> {
    public:
        inline explicit TSearcher(const TCapturingFsm& fsm)
            : TMatcherBase<TCapturingFsm>(fsm)
        {
        }

        inline bool Captured() const noexcept {
            return GetState().Captured();
        }

        inline TSearcher& Search(const char* data, size_t len, bool addBegin = true, bool addEnd = true) noexcept {
            Data = TStringBuf(data, len);
            Run(data, len, addBegin, addEnd);
            return *this;
        }

        inline TSearcher& Search(const TStringBuf& s) noexcept {
            return Search(s.data(), s.size());
        }

        inline TStringBuf GetCaptured() const noexcept {
            return TStringBuf(Data.data() + GetState().Begin() - 1,
                              Data.data() + GetState().End() - 1);
        }

    private:
        TStringBuf Data;
    };

    class TSlowSearcher : TMatcherBase<TSlowCapturingFsm>{
    public:
        typedef typename TSlowCapturingFsm::TScanner::State TState;
        inline explicit TSlowSearcher(const TSlowCapturingFsm& fsm)
                : TMatcherBase<TSlowCapturingFsm>(fsm)
                , HasCaptured(false)
        {
        }

        inline bool Captured() const noexcept {
            return HasCaptured;
        }

        inline TSlowSearcher& Search(const char* data, size_t len, bool addBegin = false, bool addEnd = false) noexcept {
            TStringBuf textData(data, len);
            Data = textData;
            Run(Data.begin(), Data.size(), addBegin, addEnd);
            return GetAns();
        }

        inline TSlowSearcher& Search(const TStringBuf& s) noexcept {
            return Search(s.data(), s.size());
        }

        inline TStringBuf GetCaptured() const noexcept {
            return Ans;
        }

    private:
        TStringBuf Data;
        TStringBuf Ans;
        bool HasCaptured;

        inline TSlowSearcher& GetAns() {
            auto state = GetState();
            Pire::SlowCapturingScanner::SingleState final;
            if (!GetScanner().GetCapture(state, final)) {
                HasCaptured = false;
            } else {
                if (!final.HasEnd()) {
                    final.SetEnd(Data.size());
                }
                Ans = TStringBuf(Data, final.GetBegin(), final.GetEnd() - final.GetBegin());
                HasCaptured = true;
            }
            return *this;
        }
    };
}
