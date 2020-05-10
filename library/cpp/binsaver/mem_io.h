#pragma once

#include "bin_saver.h"

namespace NMemIoInternals {
    class TMemoryStream: public IBinaryStream {
        TVector<char>& Data;
        ui64 Pos;

    public:
        TMemoryStream(TVector<char>* data, ui64 pos = 0)
            : Data(*data)
            , Pos(pos)
        {
        }
        ~TMemoryStream() override {
        } // keep gcc happy

        bool IsValid() const override {
            return true;
        }
        bool IsFailed() const override {
            return false;
        }

    private:
        int WriteImpl(const void* userBuffer, int size) override {
            if (size == 0)
                return 0;
            Y_ASSERT(size > 0);
            if (Pos + size > Data.size())
                Data.yresize(Pos + size);
            memcpy(&Data[Pos], userBuffer, size);
            Pos += size;
            return size;
        }
        int ReadImpl(void* userBuffer, int size) override {
            if (size == 0)
                return 0;
            Y_ASSERT(size > 0);
            int res = Min(Data.size() - Pos, (ui64)size);
            if (res)
                memcpy(userBuffer, &Data[Pos], res);
            Pos += res;
            return res;
        }
    };

    template <class T>
    inline void SerializeMem(bool bRead, TVector<char>* data, T& c, bool stableOutput = false) {
        if (IBinSaver::HasNonTrivialSerializer<T>(0u)) {
            TMemoryStream f(data);
            {
                IBinSaver bs(f, bRead, stableOutput);
                bs.Add(1, &c);
            }
        } else {
            if (bRead) {
                Y_ASSERT(data->size() == sizeof(T));
                c = *reinterpret_cast<T*>(&(*data)[0]);
            } else {
                data->yresize(sizeof(T));
                *reinterpret_cast<T*>(&(*data)[0]) = c;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    class THugeMemoryStream: public IBinaryStream {
        TVector<TVector<char>>& Data;
        i64 Block, Pos;
        bool ShrinkOnRead;

        enum {
            MAX_BLOCK_SIZE = 1024 * 1024 // Aligned with cache size
        };

    public:
        THugeMemoryStream(TVector<TVector<char>>* data, bool shrinkOnRead = false)
            : Data(*data)
            , Block(0)
            , Pos(0)
            , ShrinkOnRead(shrinkOnRead)
        {
            Y_ASSERT(!data->empty());
        }

        ~THugeMemoryStream() override {
        } // keep gcc happy

        bool IsValid() const override {
            return true;
        }
        bool IsFailed() const override {
            return false;
        }

    private:
        int WriteImpl(const void* userDataArg, int sizeArg) override {
            if (sizeArg == 0)
                return 0;
            const char* userData = (const char*)userDataArg;
            i64 size = sizeArg;
            i64 newSize = Pos + size;
            if (newSize > Data[Block].ysize()) {
                while (newSize > MAX_BLOCK_SIZE) {
                    int maxWrite = MAX_BLOCK_SIZE - Pos;
                    Data[Block].yresize(MAX_BLOCK_SIZE);
                    if (maxWrite) {
                        memcpy(&Data[Block][Pos], userData, maxWrite);
                        userData += maxWrite;
                        size -= maxWrite;
                    }
                    ++Block;
                    Pos = 0;
                    Data.resize(Block + 1);
                    newSize = Pos + size;
                }
                Data[Block].yresize(newSize);
            }
            if (size) {
                memcpy(&Data[Block][Pos], userData, size);
            }
            Pos += size;
            return sizeArg;
        }
        int ReadImpl(void* userDataArg, int sizeArg) override {
            if (sizeArg == 0)
                return 0;

            char* userData = (char*)userDataArg;
            i64 size = sizeArg;
            i64 rv = 0;
            while (size > 0) {
                int curBlockSize = Data[Block].ysize();
                int maxRead = 0;
                if (Pos + size > curBlockSize) {
                    maxRead = curBlockSize - Pos;
                    if (maxRead) {
                        memcpy(userData, &Data[Block][Pos], maxRead);
                        userData += maxRead;
                        size -= maxRead;
                        rv += maxRead;
                    }
                    if (Block + 1 == Data.ysize()) {
                        memset(userData, 0, size);
                        return rv;
                    }
                    if (ShrinkOnRead) {
                        TVector<char>().swap(Data[Block]);
                    }
                    ++Block;
                    Pos = 0;
                } else {
                    memcpy(userData, &Data[Block][Pos], size);
                    Pos += size;
                    rv += size;
                    return rv;
                }
            }
            return rv;
        }
    };

    template <class T>
    inline void SerializeMem(bool bRead, TVector<TVector<char>>* data, T& c, bool stableOutput = false) {
        if (data->empty()) {
            data->resize(1);
        }
        THugeMemoryStream f(data);
        {
            IBinSaver bs(f, bRead, stableOutput);
            bs.Add(1, &c);
        }
    }
}

template <class T>
inline void SerializeMem(const TVector<char>& data, T& c) {
    if (IBinSaver::HasNonTrivialSerializer<T>(0u)) {
        TVector<char> tmp(data);
        SerializeFromMem(&tmp, c);
    } else {
        Y_ASSERT(data.size() == sizeof(T));
        c = *reinterpret_cast<const T*>(&data[0]);
    }
}

template <class T, class D>
inline void SerializeToMem(D* data, T& c, bool stableOutput = false) {
    NMemIoInternals::SerializeMem(false, data, c, stableOutput);
}

template <class T, class D>
inline void SerializeFromMem(D* data, T& c, bool stableOutput = false) {
    NMemIoInternals::SerializeMem(true, data, c, stableOutput);
}

// Frees memory in (*data)[i] immediately upon it's deserialization, thus keeps low overall memory consumption for data + object.
template <class T>
inline void SerializeFromMemShrinkInput(TVector<TVector<char>>* data, T& c) {
    if (data->empty()) {
        data->resize(1);
    }
    NMemIoInternals::THugeMemoryStream f(data, true);
    {
        IBinSaver bs(f, true, false);
        bs.Add(1, &c);
    }
    data->resize(0);
    data->shrink_to_fit();
}
