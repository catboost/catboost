#pragma once

#include <library/cpp/http/io/stream.h>
#include <util/generic/ptr.h>

class TSocket;

/// Потоки ввода/вывода для получения запросов и отправки ответов HTTP-сервера.
class THttpServerConn {
public:
    explicit THttpServerConn(const TSocket& s);
    THttpServerConn(const TSocket& s, size_t outputBufferSize);
    ~THttpServerConn();

    THttpInput* Input() noexcept;
    THttpOutput* Output() noexcept;

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
    class TImpl;
    THolder<TImpl> Impl_;
};
