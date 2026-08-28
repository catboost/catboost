#pragma once

#include "headers.h"

#include <util/stream/output.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/strbuf.h>
#include <util/generic/yexception.h>
#include <util/generic/array_ref.h>

class TSocket;

struct THttpException: public yexception {
};

struct THttpParseException: public THttpException {
};

struct THttpReadException: public THttpException {
};

// Body ended before Content-Length. Thrown only under TOptions::StrictContentLength.
struct THttpTruncatedBodyException: public THttpReadException {
};

/// Чтение ответа HTTP-сервера.
class THttpInput: public IInputStream {
public:
    struct TOptions {
        // If Content-Length is present, throw THttpTruncatedBodyException once the underlying
        // stream reaches EOF with fewer bytes read than announced. A caller that stops reading
        // early and destroys the stream does not trigger it.
        // Do not enable for HEAD responses: they announce Content-Length but carry no body,
        // which is indistinguishable from truncation at this level.
        bool StrictContentLength = false;
    };

    THttpInput(IInputStream* slave);
    THttpInput(IInputStream* slave, const TOptions& options);
    THttpInput(THttpInput&& httpInput);
    ~THttpInput() override;

    /*
     * parsed http headers
     */
    /// Возвращает контейнер с заголовками ответа HTTP-сервера.
    const THttpHeaders& Headers() const noexcept;

    /*
     * parsed http trailers
     */
    /// Возвращает контейнер (возможно пустой) с trailer'ами ответа HTTP-сервера.
    /// Поток должен быть вычитан полностью прежде чем trailer'ы будут доступны.
    /// Пока поток не вычитан до конца возвращается Nothing.
    /// https://tools.ietf.org/html/rfc7230#section-4.1.2
    const TMaybe<THttpHeaders>& Trailers() const noexcept;

    /*
     * first line - response or request
     */
    /// Возвращает первую строку ответа HTTP-сервера.
    /// @details Первая строка HTTP-сервера - строка состояния,
    /// содержащая три поля: версию HTTP, код состояния и описание.
    const TString& FirstLine() const noexcept;

    /*
     * connection can be keep-alive
     */
    /// Проверяет, не завершено ли соединение с сервером.
    /// @details Транзакция считается завершенной, если не передан заголовок
    /// "Connection: Keep Alive".
    bool IsKeepAlive() const noexcept;

    /*
     * output data can be encoded
     */
    /// Проверяет, поддерживается ли данный тип кодирования содержимого
    /// ответа HTTP-сервера.
    bool AcceptEncoding(const TString& coding) const;

    /// Пытается определить наилучший тип кодирования ответа HTTP-сервера.
    /// @details Если ответ сервера говорит о том, что поддерживаются
    /// любые типы кодирования, выбирается gzip. В противном случае
    /// из списка типов кодирования выбирается лучший из поддерживаемых сервером.
    TString BestCompressionScheme() const;
    TString BestCompressionScheme(TArrayRef<const TStringBuf> codings) const;

    /// Если заголовки содержат Content-Length, возвращает true и
    /// записывает значение из заголовка в value
    bool GetContentLength(ui64& value) const noexcept;

    /// Признак запакованности данных, - если выставлен, то Content-Length, при наличии в заголовках,
    /// показывает объём запакованных данных, а из THttpInput мы будем вычитывать уже распакованные.
    bool ContentEncoded() const noexcept;

    /// Сколько байт из заявленных в Content-Length ещё не вычитано из тела (до распаковки).
    /// Всегда 0, если Content-Length в ответе нет или используется chunked encoding.
    /// После полного вычитывания тела ненулевое значение означает обрыв ответа.
    ui64 ContentLengthLeft() const noexcept;

    /// Returns true if Content-Length or Transfer-Encoding header received
    bool HasContent() const noexcept;

    bool HasExpect100Continue() const noexcept;

private:
    size_t DoRead(void* buf, size_t len) override;
    size_t DoSkip(size_t len) override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

/// Передача запроса HTTP-серверу.
class THttpOutput: public IOutputStream {
public:
    THttpOutput(IOutputStream* slave);
    THttpOutput(IOutputStream* slave, THttpInput* request);
    ~THttpOutput() override;

    /*
     * sent http headers
     */
    /// Возвращает контейнер с заголовками запроса к HTTP-серверу.
    const THttpHeaders& SentHeaders() const noexcept;

    /// Устанавливает режим, при котором сервер выдает ответ в упакованном виде.
    void EnableCompression(bool enable);
    void EnableCompression(TArrayRef<const TStringBuf> schemas);

    /// Устанавливает режим, при котором соединение с сервером не завершается
    /// после окончания транзакции.
    void EnableKeepAlive(bool enable);

    /// Устанавливает режим, при котором тело HTTP-запроса/ответа преобразуется в соответствии
    /// с заголовками Content-Encoding и Transfer-Encoding (включен по умолчанию)
    void EnableBodyEncoding(bool enable);

    /// Устанавливает режим, при котором тело HTTP-ответа сжимается кодеком
    /// указанным в Content-Encoding (включен по умолчанию)
    void EnableCompressionHeader(bool enable);

    /// Проверяет, производится ли выдача ответов в упакованном виде.
    bool IsCompressionEnabled() const noexcept;

    /// Проверяет, не завершается ли соединение с сервером после окончания транзакции.
    bool IsKeepAliveEnabled() const noexcept;

    /// Проверяет, преобразуется ли тело HTTP-запроса/ответа в соответствии
    /// с заголовками Content-Encoding и Transfer-Encoding
    bool IsBodyEncodingEnabled() const noexcept;

    /// Проверяет, сжимается ли тело HTTP-ответа кодеком
    /// указанным в Content-Encoding
    bool IsCompressionHeaderEnabled() const noexcept;

    /*
     * is this connection can be really keep-alive
     */
    /// Проверяет, можно ли установить режим, при котором соединение с сервером
    /// не завершается после окончания транзакции.
    bool CanBeKeepAlive() const noexcept;

    void SendContinue();

    /*
     * first line - response or request
     */
    /// Возвращает первую строку HTTP-запроса/ответа
    const TString& FirstLine() const noexcept;

    /// Возвращает размер отправленных данных (без заголовков, с учётом сжатия, без
    /// учёта chunked transfer encoding)
    size_t SentSize() const noexcept;

private:
    void DoWrite(const void* buf, size_t len) override;
    void DoFlush() override;
    void DoFinish() override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

/// Возвращает код состояния из ответа сервера.
unsigned ParseHttpRetCode(const TStringBuf& ret);

/// Отправляет HTTP-серверу запрос с минимумом необходимых заголовков.
void SendMinimalHttpRequest(TSocket& s, const TStringBuf& host, const TStringBuf& request, const TStringBuf& agent = "YandexSomething/1.0", const TStringBuf& from = "webadmin@yandex.ru");

TArrayRef<const TStringBuf> SupportedCodings();

/// @}
