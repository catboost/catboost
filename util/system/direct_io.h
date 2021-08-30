#pragma once

#include "align.h"

#include "file.h"
#include <util/generic/buffer.h>

// Supports Linux Direct-IO:
// - Simple buffering logic.
// - Default buffer size of 128KB matches VM page writeback granularity, to maximize IO throughput.
// - Supports writing odd sized files by turning off direct IO for the last chunk.
class TDirectIOBufferedFile {
public:
    TDirectIOBufferedFile(const TString& path, EOpenMode oMode, size_t buflen = 1 << 17);
    ~TDirectIOBufferedFile();

    void FlushData();
    void Finish();
    size_t Read(void* buffer, size_t byteCount);
    void Write(const void* buffer, size_t byteCount);
    size_t Pread(void* buffer, size_t byteCount, ui64 offset);
    void Pwrite(const void* buffer, size_t byteCount, ui64 offset);

    inline bool IsOpen() const {
        return true;
    }

    inline ui64 GetWritePosition() const {
        return WritePosition;
    }

    inline ui64 GetLength() const {
        return FlushedBytes + DataLen;
    }

    inline FHANDLE GetHandle() {
        return File.GetHandle();
    }

    inline void FallocateNoResize(ui64 length) {
        File.FallocateNoResize(length);
    }

    inline void ShrinkToFit() {
        File.ShrinkToFit();
    }

private:
    inline bool IsAligned(i64 value) {
        return Alignment ? value == AlignDown<i64>(value, Alignment) : true;
    }

    inline bool IsAligned(const void* value) {
        return Alignment ? value == AlignDown(value, Alignment) : true;
    }

    size_t PreadSafe(void* buffer, size_t byteCount, ui64 offset);
    size_t ReadFromFile(void* buffer, size_t byteCount, ui64 offset);
    void WriteToFile(const void* buf, size_t len, ui64 position);
    void WriteToBuffer(const void* buf, size_t len, ui64 position);
    void SetDirectIO(bool value);

private:
    TFile File;
    size_t Alignment;
    size_t BufLen;
    size_t DataLen;
    void* Buffer;
    TBuffer BufferStorage;
    ui64 ReadPosition;
    ui64 WritePosition;
    ui64 FlushedBytes;
    ui64 FlushedToDisk;
    bool DirectIO;
};
