#include <util/system/defaults.h>
#include <util/system/yassert.h>
#include <library/cpp/charset/codepage.h>
#include <util/generic/singleton.h>
#include <util/generic/yexception.h>
#include <library/cpp/charset/doccodes.h>

#include "pire.h"

namespace NPire {
    namespace {
        // A one-byte encoding which is capable of transforming upper half of the character
        // table to/from Unicode chars.
        class TOneByte: public TEncoding {
        public:
            TOneByte(ECharset doccode) {
                Table_ = CodePageByCharset(doccode)->unicode;
                for (size_t i = 0; i < 256; ++i)
                    Reverse_.insert(std::make_pair(Table_[i], static_cast<char>(i)));
            }

            wchar32 FromLocal(const char*& begin, const char* end) const override {
                if (begin != end)
                    return Table_[static_cast<unsigned char>(*begin++)];
                else
                    ythrow yexception() << "EOF reached in Pire::OneByte::fromLocal()";
            }

            TString ToLocal(wchar32 c) const override {
                THashMap<wchar32, char>::const_iterator i = Reverse_.find(c);
                if (i != Reverse_.end())
                    return TString(1, i->second);
                else
                    return TString();
            }

            void AppendDot(TFsm& fsm) const override {
                fsm.AppendDot();
            }

        private:
            const wchar32* Table_;
            THashMap<wchar32, char> Reverse_;
        };

        template <unsigned N>
        struct TOneByteHelper: public TOneByte {
            inline TOneByteHelper()
                : TOneByte((ECharset)N)
            {
            }
        };
    }

    namespace NEncodings {
        const NPire::TEncoding& Koi8r() {
            return *Singleton<TOneByteHelper<CODES_KOI8>>();
        }

        const NPire::TEncoding& Cp1251() {
            return *Singleton<TOneByteHelper<CODES_WIN>>();
        }

        const NPire::TEncoding& Get(ECharset encoding) {
            switch (encoding) {
                case CODES_WIN:
                    return Cp1251();
                case CODES_KOI8:
                    return Koi8r();
                case CODES_ASCII:
                    return NPire::NEncodings::Latin1();
                case CODES_UTF8:
                    return NPire::NEncodings::Utf8();
                default:
                    ythrow yexception() << "Pire::Encodings::get(ECharset): unknown encoding " << (int)encoding;
            }
        }

    }

}
