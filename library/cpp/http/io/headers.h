#pragma once

#include <util/generic/array_ref.h>
#include <util/generic/deque.h>
#include <util/generic/string.h>
#include <util/generic/strbuf.h>
#include <util/generic/vector.h>  // XXX unused - remove after fixing transitive includes.
#include <util/string/cast.h>

class IInputStream;
class IOutputStream;

/// @addtogroup Streams_HTTP
/// @{
/// Объект, содержащий информацию о HTTP-заголовке.
class THttpInputHeader {
public:
    THttpInputHeader() = delete;
    THttpInputHeader(const THttpInputHeader&) = default;
    THttpInputHeader(THttpInputHeader&&) = default;
    THttpInputHeader& operator=(const THttpInputHeader&) = default;
    THttpInputHeader& operator=(THttpInputHeader&&) = default;

    /// @param[in] header - строка вида 'параметр: значение'.
    THttpInputHeader(TStringBuf header);
    /// @param[in] name - имя параметра.
    /// @param[in] value - значение параметра.
    THttpInputHeader(TString name, TString value);

    /// Возвращает имя параметра.
    inline const TString& Name() const noexcept {
        return Name_;
    }

    /// Возвращает значение параметра.
    inline const TString& Value() const noexcept {
        return Value_;
    }

    /// Записывает заголовок вида "имя параметра: значение\r\n" в поток.
    void OutTo(IOutputStream* stream) const;

    /// Возвращает строку "имя параметра: значение".
    inline TString ToString() const {
        return Name_ + TStringBuf(": ") + Value_;
    }

private:
    TString Name_;
    TString Value_;
};

/// Контейнер для хранения HTTP-заголовков
class THttpHeaders {
    using THeaders = TDeque<THttpInputHeader>;

public:
    using TConstIterator = THeaders::const_iterator;

    THttpHeaders() = default;
    THttpHeaders(const THttpHeaders&) = default;
    THttpHeaders& operator=(const THttpHeaders&) = default;
    THttpHeaders(THttpHeaders&&) = default;
    THttpHeaders& operator=(THttpHeaders&&) = default;

    /// Добавляет каждую строку из потока в контейнер, считая ее правильным заголовком.
    THttpHeaders(IInputStream* stream);

    /// Создаёт контейнер из initializer-list'а или массива/вектора хедеров.
    /// Пример: `THttpHeaders headers({{"Host", "example.com"}});`
    THttpHeaders(TArrayRef<const THttpInputHeader> headers);

    /// Стандартный итератор.
    inline TConstIterator Begin() const noexcept {
        return Headers_.begin();
    }
    inline TConstIterator begin() const noexcept {
        return Headers_.begin();
    }

    /// Стандартный итератор.
    inline TConstIterator End() const noexcept {
        return Headers_.end();
    }
    inline TConstIterator end() const noexcept {
        return Headers_.end();
    }

    /// Возвращает количество заголовков в контейнере.
    inline size_t Count() const noexcept {
        return Headers_.size();
    }

    /// Проверяет, содержит ли контейнер хотя бы один заголовок.
    inline bool Empty() const noexcept {
        return Headers_.empty();
    }

    /// Добавляет заголовок в контейнер.
    void AddHeader(THttpInputHeader header);

    template <typename ValueType>
    void AddHeader(TString name, const ValueType& value) {
        AddHeader(THttpInputHeader(std::move(name), ToString(value)));
    }

    /// Добавляет заголовок в контейнер, если тот не содержит заголовка
    /// c таким же параметром. В противном случае, заменяет существующий
    /// заголовок на новый.
    void AddOrReplaceHeader(const THttpInputHeader& header);

    template <typename ValueType>
    void AddOrReplaceHeader(TString name, const ValueType& value) {
        AddOrReplaceHeader(THttpInputHeader(std::move(name), ToString(value)));
    }

    // Проверяет, есть ли такой заголовок
    bool HasHeader(TStringBuf header) const;

    /// Удаляет заголовок, если он есть.
    void RemoveHeader(TStringBuf header);

    /// Ищет заголовок по указанному имени
    /// Возвращает nullptr, если не нашел
    const THttpInputHeader* FindHeader(TStringBuf header) const;

    /// Записывает все заголовки контейнера в поток.
    /// @details Каждый заголовк записывается в виде "имя параметра: значение\r\n".
    void OutTo(IOutputStream* stream) const;

    /// Обменивает наборы заголовков двух контейнеров.
    void Swap(THttpHeaders& headers) noexcept {
        Headers_.swap(headers.Headers_);
    }

private:
    THeaders Headers_;
};

/// @}
