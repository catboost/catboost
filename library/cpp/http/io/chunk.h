#pragma once

#include <util/stream/output.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>

class THttpHeaders;

/// @addtogroup Streams_Chunked
/// @{
/// Ввод данных порциями.
/// @details Последовательное чтение блоков данных. Предполагается, что
/// данные записаны в виде <длина блока><блок данных>.
class TChunkedInput: public IInputStream {
public:
    /// Если передан указатель на trailers, то туда будут записаны HTTP trailer'ы (возможно пустые),
    /// которые идут после чанков.
    TChunkedInput(IInputStream* slave, TMaybe<THttpHeaders>* trailers = nullptr);
    ~TChunkedInput() override;

private:
    size_t DoRead(void* buf, size_t len) override;
    size_t DoSkip(size_t len) override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

/// Вывод данных порциями.
/// @details Вывод данных блоками в виде <длина блока><блок данных>. Если объем
/// данных превышает 64K, они записываются в виде n блоков по 64K + то, что осталось.
class TChunkedOutput: public IOutputStream {
public:
    TChunkedOutput(IOutputStream* slave);
    ~TChunkedOutput() override;

protected:
    void DoWrite(const void* buf, size_t len) override;

private:
    void DoFlush() override;
    void DoFinish() override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};
/// @}
