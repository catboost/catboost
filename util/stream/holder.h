#pragma once

#include <util/generic/ptr.h>

#include <utility>
#include <type_traits>

class TInputStream;
class TOutputStream;

namespace NPrivate {
    template <class Stream, bool isInput = std::is_base_of<TInputStream, Stream>::value>
    struct TStreamBase {
        using TType = TInputStream;
    };

    template <class Stream>
    struct TStreamBase<Stream, false> {
        using TType = TOutputStream;
    };

} // namespace NPrivate

/**
 * An ownership-gaining wrapper for proxy streams.
 *
 * Example usage:
 * \code
 * TCountingInput* input = new THoldingStream<TCountingInput>(new TStringInput(s));
 * \encode
 *
 * In this example, resulting counting input also owns a string input that it
 * was constructed on top of.
 */
template <class Base, class StreamBase = typename ::NPrivate::TStreamBase<Base>::TType>
class THoldingStream: private THolder<StreamBase>, public Base {
public:
    template <class... Args>
    inline THoldingStream(StreamBase* stream, Args&&... args)
        : THolder<StreamBase>(stream)
        , Base(this->Get(), std::forward<Args>(args)...)
    {
    }
};
