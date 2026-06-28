#pragma once

#include <library/cpp/http/io/stream.h>
#include <util/generic/ptr.h>
#include <util/stream/buffered.h>

class IInputStream;
class IOutputStream;
class TSocket;

/// Потоки ввода/вывода для получения запросов и отправки ответов HTTP-сервера.
class THttpServerConn {
public:
    class ISocketStreams {
    public:
        virtual ~ISocketStreams() {}

        virtual IInputStream* Input() = 0;
        virtual IOutputStream* Output() = 0;
        virtual void Reset() = 0;
    };
public:
    THttpServerConn(const TSocket& s);
    THttpServerConn(const TSocket& s, size_t outputBufferSize);
    THttpServerConn(THolder<ISocketStreams> socketStreams, size_t outputBufferSize);

    ~THttpServerConn();

    THttpInput* Input() noexcept {
        return &HttpInput_;
    }

    THttpOutput* Output() noexcept {
        return &HttpOutput_;
    }

    inline const THttpInput* Input() const noexcept {
        return const_cast<THttpServerConn*>(this)->Input();
    }

    inline const THttpOutput* Output() const noexcept {
        return const_cast<THttpServerConn*>(this)->Output();
    }

    /// Проверяет, можно ли установить режим, при котором соединение с сервером
    /// не завершается после окончания транзакции.
    inline bool CanBeKeepAlive() const noexcept {
        return Output()->CanBeKeepAlive();
    }

    void Reset();
private:
    THolder<ISocketStreams> SocketStreams_;

    TBufferedOutput BufferedOutput_;
    THttpInput HttpInput_;
    THttpOutput HttpOutput_;
};
