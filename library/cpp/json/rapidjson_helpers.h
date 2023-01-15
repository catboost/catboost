#pragma once

#include <util/generic/strbuf.h>
#include <util/stream/input.h>

namespace NJson {
    struct TReadOnlyStreamBase {
        using Ch = char;

        Ch* PutBegin() {
            Y_ASSERT(false);
            return nullptr;
        }

        void Put(Ch) {
            Y_ASSERT(false);
        }

        void Flush() {
            Y_ASSERT(false);
        }

        size_t PutEnd(Ch*) {
            Y_ASSERT(false);
            return 0;
        }
    };

    struct TInputStreamWrapper : TReadOnlyStreamBase {
        Ch Peek() const {
            if (!Eof) {
                if (Pos >= Sz) {
                    if (Sz < BUF_SIZE) {
                        Sz += Helper.Read(Buf + Sz, BUF_SIZE - Sz);
                    } else {
                        Sz = Helper.Read(Buf, BUF_SIZE);
                        Pos = 0;
                    }
                }

                if (Pos < Sz) {
                    return Buf[Pos];
                }
            }

            Eof = true;
            return 0;
        }

        Ch Take() {
            auto c = Peek();
            ++Pos;
            ++Count;
            return c;
        }

        size_t Tell() const {
            return Count;
        }

        TInputStreamWrapper(IInputStream& helper)
            : Helper(helper)
            , Eof(false)
            , Sz(0)
            , Pos(0)
            , Count(0)
        {
        }

        static const size_t BUF_SIZE = 1 << 12;

        IInputStream& Helper;
        mutable char Buf[BUF_SIZE];
        mutable bool Eof;
        mutable size_t Sz;
        mutable size_t Pos;
        size_t Count;
    };

    struct TStringBufStreamWrapper : TReadOnlyStreamBase {
        Ch Peek() const {
            return Pos < Data.size() ? Data[Pos] : 0;
        }

        Ch Take() {
            auto c = Peek();
            ++Pos;
            return c;
        }

        size_t Tell() const {
            return Pos;
        }

        TStringBufStreamWrapper(TStringBuf data)
            : Data(data)
            , Pos(0)
        {
        }

        TStringBuf Data;
        size_t Pos;
    };
}
