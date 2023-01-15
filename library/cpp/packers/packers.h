#pragma once

#include <util/generic/string.h>
#include <util/generic/strbuf.h>
#include <util/generic/set.h>
#include <util/generic/list.h>
#include <util/generic/vector.h>
#include <util/generic/bitops.h>

#include <array>
// Data serialization strategy class.
// Default realization can pack only limited range of types, but you can pack any data other using your own strategy class.

template <class T>
class TNullPacker { // Very effective package class - pack any data into zero bytes :)
public:
    void UnpackLeaf(const char*, T& t) const {
        t = T();
    }

    void PackLeaf(char*, const T&, size_t) const {
    }

    size_t MeasureLeaf(const T&) const {
        return 0;
    }

    size_t SkipLeaf(const char*) const {
        return 0;
    }
};

template <typename T>
class TAsIsPacker { // this packer is not really a packer...
public:
    void UnpackLeaf(const char* p, T& t) const {
        memcpy(&t, p, sizeof(T));
    }
    void PackLeaf(char* buffer, const T& data, size_t computedSize) const {
        Y_ASSERT(computedSize == sizeof(data));
        memcpy(buffer, &data, sizeof(T));
    }
    size_t MeasureLeaf(const T& data) const {
        Y_UNUSED(data);
        return sizeof(T);
    }
    size_t SkipLeaf(const char*) const {
        return sizeof(T);
    }
};

// Implementation

namespace NPackers {
    template <class T>
    inline ui64 ConvertIntegral(const T& data);

    template <>
    inline ui64 ConvertIntegral(const i64& data) {
        if (data < 0) {
            return (static_cast<ui64>(-1 * data) << 1) | 1;
        } else {
            return static_cast<ui64>(data) << 1;
        }
    }

    namespace NImpl {
        template <class T, bool isSigned>
        struct TConvertImpl {
            static inline ui64 Convert(const T& data);
        };

        template <class T>
        struct TConvertImpl<T, true> {
            static inline ui64 Convert(const T& data) {
                return ConvertIntegral<i64>(static_cast<i64>(data));
            }
        };

        template <class T>
        struct TConvertImpl<T, false> {
            static inline ui64 Convert(const T& data) {
                return data;
            }
        };
    }

    template <class T>
    inline ui64 ConvertIntegral(const T& data) {
        static_assert(std::is_integral<T>::value, "T must be integral type");
        return NImpl::TConvertImpl<T, std::is_signed<T>::value>::Convert(data);
    }

    //---------------------------------
    // TIntegralPacker --- for integral types.

    template <class T>
    class TIntegralPacker { // can pack only integral types <= ui64
    public:
        void UnpackLeaf(const char* p, T& t) const;
        void PackLeaf(char* buffer, const T& data, size_t size) const;
        size_t MeasureLeaf(const T& data) const;
        size_t SkipLeaf(const char* p) const;
    };

    template <>
    inline size_t TIntegralPacker<ui64>::MeasureLeaf(const ui64& val) const {
        constexpr size_t MAX_SIZE = sizeof(ui64) + sizeof(ui64) / 8;

        ui64 value = val;
        size_t len = 1;

        value >>= 7;
        for (; value && len < MAX_SIZE; value >>= 7)
            ++len;

        return len;
    }

    template <>
    inline void TIntegralPacker<ui64>::PackLeaf(char* buffer, const ui64& val, size_t len) const {
        ui64 value = val;
        int lenmask = 0;

        for (size_t i = len - 1; i; --i) {
            buffer[i] = (char)(value & 0xFF);
            value >>= 8;
            lenmask = ((lenmask >> 1) | (1 << 7));
        }

        buffer[0] = (char)(lenmask | value);
    }

    extern const ui8 SkipTable[];

    template <>
    inline void TIntegralPacker<ui64>::UnpackLeaf(const char* p, ui64& result) const {
        unsigned char ch = *(p++);
        size_t taillen = SkipTable[ch] - 1;

        result = (ch & (0x7F >> taillen));

        while (taillen--)
            result = ((result << 8) | (*(p++) & 0xFF));
    }

    template <>
    inline size_t TIntegralPacker<ui64>::SkipLeaf(const char* p) const {
        return SkipTable[(ui8)*p];
    }

    namespace NImpl {
        template <class T, bool isSigned>
        struct TUnpackLeafImpl {
            inline void UnpackLeaf(const char* p, T& t) const;
        };
        template <class T>
        struct TUnpackLeafImpl<T, true> {
            inline void UnpackLeaf(const char* p, T& t) const {
                ui64 val;
                TIntegralPacker<ui64>().UnpackLeaf(p, val);
                if (val & 1) {
                    t = -1 * static_cast<i64>(val >> 1);
                } else {
                    t = static_cast<T>(val >> 1);
                }
            }
        };
        template <class T>
        struct TUnpackLeafImpl<T, false> {
            inline void UnpackLeaf(const char* p, T& t) const {
                ui64 tmp;
                TIntegralPacker<ui64>().UnpackLeaf(p, tmp);
                t = static_cast<T>(tmp);
            }
        };
    }

    template <class T>
    inline void TIntegralPacker<T>::UnpackLeaf(const char* p, T& t) const {
        NImpl::TUnpackLeafImpl<T, std::is_signed<T>::value>().UnpackLeaf(p, t);
    }

    template <class T>
    inline void TIntegralPacker<T>::PackLeaf(char* buffer, const T& data, size_t size) const {
        TIntegralPacker<ui64>().PackLeaf(buffer, ConvertIntegral<T>(data), size);
    }

    template <class T>
    inline size_t TIntegralPacker<T>::MeasureLeaf(const T& data) const {
        return TIntegralPacker<ui64>().MeasureLeaf(ConvertIntegral<T>(data));
    }

    template <class T>
    inline size_t TIntegralPacker<T>::SkipLeaf(const char* p) const {
        return TIntegralPacker<ui64>().SkipLeaf(p);
    }

    //-------------------------------------------
    // TFPPacker --- for float/double
    namespace NImpl {
        template <class TFloat, class TUInt>
        class TFPPackerBase {
        protected:
            typedef TIntegralPacker<TUInt> TPacker;

            union THelper {
                TFloat F;
                TUInt U;
            };

            TFloat FromUInt(TUInt u) const {
                THelper h;
                h.U = ReverseBytes(u);
                return h.F;
            }

            TUInt ToUInt(TFloat f) const {
                THelper h;
                h.F = f;
                return ReverseBytes(h.U);
            }

        public:
            void UnpackLeaf(const char* c, TFloat& t) const {
                TUInt u = 0;
                TPacker().UnpackLeaf(c, u);
                t = FromUInt(u);
            }

            void PackLeaf(char* c, const TFloat& t, size_t sz) const {
                TPacker().PackLeaf(c, ToUInt(t), sz);
            }

            size_t MeasureLeaf(const TFloat& t) const {
                return TPacker().MeasureLeaf(ToUInt(t));
            }

            size_t SkipLeaf(const char* c) const {
                return TPacker().SkipLeaf(c);
            }
        };
    }

    class TFloatPacker: public NImpl::TFPPackerBase<float, ui32> {
    };

    class TDoublePacker: public NImpl::TFPPackerBase<double, ui64> {
    };

    //-------------------------------------------
    // TStringPacker --- for TString/TUtf16String and TStringBuf.

    template <class TStringType>
    class TStringPacker {
    public:
        void UnpackLeaf(const char* p, TStringType& t) const;
        void PackLeaf(char* buffer, const TStringType& data, size_t size) const;
        size_t MeasureLeaf(const TStringType& data) const;
        size_t SkipLeaf(const char* p) const;
    };

    template <class TStringType>
    inline void TStringPacker<TStringType>::UnpackLeaf(const char* buf, TStringType& t) const {
        size_t len;
        TIntegralPacker<size_t>().UnpackLeaf(buf, len);
        size_t start = TIntegralPacker<size_t>().SkipLeaf(buf);
        t = TStringType((const typename TStringType::char_type*)(buf + start), len);
    }

    template <class TStringType>
    inline void TStringPacker<TStringType>::PackLeaf(char* buf, const TStringType& str, size_t size) const {
        size_t len = str.size();
        size_t lenChar = len * sizeof(typename TStringType::char_type);
        size_t start = size - lenChar;
        TIntegralPacker<size_t>().PackLeaf(buf, len, TIntegralPacker<size_t>().MeasureLeaf(len));
        memcpy(buf + start, str.data(), lenChar);
    }

    template <class TStringType>
    inline size_t TStringPacker<TStringType>::MeasureLeaf(const TStringType& str) const {
        size_t len = str.size();
        return TIntegralPacker<size_t>().MeasureLeaf(len) + len * sizeof(typename TStringType::char_type);
    }

    template <class TStringType>
    inline size_t TStringPacker<TStringType>::SkipLeaf(const char* buf) const {
        size_t result = TIntegralPacker<size_t>().SkipLeaf(buf);
        {
            size_t len;
            TIntegralPacker<size_t>().UnpackLeaf(buf, len);
            result += len * sizeof(typename TStringType::char_type);
        }
        return result;
    }

    template <class T>
    class TPacker;

    // TContainerPacker --- for any container
    // Requirements to class C:
    //    - has method size() (returns size_t)
    //    - has subclass C::value_type
    //    - has subclass C::const_iterator
    //    - has methods begin() and end() (return C::const_iterator)
    //    - has method insert(C::const_iterator, const C::value_type&)
    //  Examples: TVector, TList, TSet
    //  Requirements to class EP: has methods as in any packer (UnpackLeaf, PackLeaf, MeasureLeaf, SkipLeaf) that
    //    are applicable to C::value_type

    template <typename T>
    struct TContainerInfo {
        enum {
            IsVector = 0
        };
    };

    template <typename T>
    struct TContainerInfo<std::vector<T>> {
        enum {
            IsVector = 1
        };
    };

    template <typename T>
    struct TContainerInfo<TVector<T>> {
        enum {
            IsVector = 1
        };
    };

    template <bool IsVector>
    class TContainerPackerHelper {
    };

    template <>
    class TContainerPackerHelper<false> {
    public:
        template <class Packer, class Container>
        static void UnpackLeaf(Packer& p, const char* buffer, Container& c) {
            p.UnpackLeafSimple(buffer, c);
        }
    };

    template <>
    class TContainerPackerHelper<true> {
    public:
        template <class Packer, class Container>
        static void UnpackLeaf(Packer& p, const char* buffer, Container& c) {
            p.UnpackLeafVector(buffer, c);
        }
    };

    template <class C, class EP = TPacker<typename C::value_type>>
    class TContainerPacker {
    private:
        typedef C TContainer;
        typedef EP TElementPacker;
        typedef typename TContainer::const_iterator TElementIterator;

        void UnpackLeafSimple(const char* buffer, TContainer& c) const;
        void UnpackLeafVector(const char* buffer, TContainer& c) const;

        friend class TContainerPackerHelper<TContainerInfo<C>::IsVector>;

    public:
        void UnpackLeaf(const char* buffer, TContainer& c) const {
            TContainerPackerHelper<TContainerInfo<C>::IsVector>::UnpackLeaf(*this, buffer, c);
        }
        void PackLeaf(char* buffer, const TContainer& data, size_t size) const;
        size_t MeasureLeaf(const TContainer& data) const;
        size_t SkipLeaf(const char* buffer) const;
    };

    template <class C, class EP>
    inline void TContainerPacker<C, EP>::UnpackLeafSimple(const char* buffer, C& result) const {
        size_t offset = TIntegralPacker<size_t>().SkipLeaf(buffer); // first value is the total size (not needed here)
        size_t len;
        TIntegralPacker<size_t>().UnpackLeaf(buffer + offset, len);
        offset += TIntegralPacker<size_t>().SkipLeaf(buffer + offset);

        result.clear();

        typename C::value_type value;
        for (size_t i = 0; i < len; i++) {
            TElementPacker().UnpackLeaf(buffer + offset, value);
            result.insert(result.end(), value);
            offset += TElementPacker().SkipLeaf(buffer + offset);
        }
    }

    template <class C, class EP>
    inline void TContainerPacker<C, EP>::UnpackLeafVector(const char* buffer, C& result) const {
        size_t offset = TIntegralPacker<size_t>().SkipLeaf(buffer); // first value is the total size (not needed here)
        size_t len;
        TIntegralPacker<size_t>().UnpackLeaf(buffer + offset, len);
        offset += TIntegralPacker<size_t>().SkipLeaf(buffer + offset);
        result.resize(len);

        for (size_t i = 0; i < len; i++) {
            TElementPacker().UnpackLeaf(buffer + offset, result[i]);
            offset += TElementPacker().SkipLeaf(buffer + offset);
        }
    }

    template <class C, class EP>
    inline void TContainerPacker<C, EP>::PackLeaf(char* buffer, const C& data, size_t size) const {
        size_t sizeOfSize = TIntegralPacker<size_t>().MeasureLeaf(size);
        TIntegralPacker<size_t>().PackLeaf(buffer, size, sizeOfSize);
        size_t len = data.size();
        size_t curSize = TIntegralPacker<size_t>().MeasureLeaf(len);
        TIntegralPacker<size_t>().PackLeaf(buffer + sizeOfSize, len, curSize);
        curSize += sizeOfSize;
        for (TElementIterator p = data.begin(); p != data.end(); p++) {
            size_t sizeChange = TElementPacker().MeasureLeaf(*p);
            TElementPacker().PackLeaf(buffer + curSize, *p, sizeChange);
            curSize += sizeChange;
        }
        Y_ASSERT(curSize == size);
    }

    template <class C, class EP>
    inline size_t TContainerPacker<C, EP>::MeasureLeaf(const C& data) const {
        size_t curSize = TIntegralPacker<size_t>().MeasureLeaf(data.size());
        for (TElementIterator p = data.begin(); p != data.end(); p++)
            curSize += TElementPacker().MeasureLeaf(*p);
        size_t extraSize = TIntegralPacker<size_t>().MeasureLeaf(curSize);

        // Double measurement protects against sudden increases in extraSize,
        // e.g. when curSize is 127 and stays in one byte, but curSize + 1 requires two bytes.

        extraSize = TIntegralPacker<size_t>().MeasureLeaf(curSize + extraSize);
        Y_ASSERT(extraSize == TIntegralPacker<size_t>().MeasureLeaf(curSize + extraSize));
        return curSize + extraSize;
    }

    template <class C, class EP>
    inline size_t TContainerPacker<C, EP>::SkipLeaf(const char* buffer) const {
        size_t value;
        TIntegralPacker<size_t>().UnpackLeaf(buffer, value);
        return value;
    }

    // TPairPacker --- for std::pair<T1, T2> (any two types; can be nested)
    // TPacker<T1> and TPacker<T2> should be valid classes

    template <class T1, class T2, class TPacker1 = TPacker<T1>, class TPacker2 = TPacker<T2>>
    class TPairPacker {
    private:
        typedef std::pair<T1, T2> TMyPair;

    public:
        void UnpackLeaf(const char* buffer, TMyPair& pair) const;
        void PackLeaf(char* buffer, const TMyPair& data, size_t size) const;
        size_t MeasureLeaf(const TMyPair& data) const;
        size_t SkipLeaf(const char* buffer) const;
    };

    template <class T1, class T2, class TPacker1, class TPacker2>
    inline void TPairPacker<T1, T2, TPacker1, TPacker2>::UnpackLeaf(const char* buffer, std::pair<T1, T2>& pair) const {
        TPacker1().UnpackLeaf(buffer, pair.first);
        size_t size = TPacker1().SkipLeaf(buffer);
        TPacker2().UnpackLeaf(buffer + size, pair.second);
    }

    template <class T1, class T2, class TPacker1, class TPacker2>
    inline void TPairPacker<T1, T2, TPacker1, TPacker2>::PackLeaf(char* buffer, const std::pair<T1, T2>& data, size_t size) const {
        size_t size1 = TPacker1().MeasureLeaf(data.first);
        TPacker1().PackLeaf(buffer, data.first, size1);
        size_t size2 = TPacker2().MeasureLeaf(data.second);
        TPacker2().PackLeaf(buffer + size1, data.second, size2);
        Y_ASSERT(size == size1 + size2);
    }

    template <class T1, class T2, class TPacker1, class TPacker2>
    inline size_t TPairPacker<T1, T2, TPacker1, TPacker2>::MeasureLeaf(const std::pair<T1, T2>& data) const {
        size_t size1 = TPacker1().MeasureLeaf(data.first);
        size_t size2 = TPacker2().MeasureLeaf(data.second);
        return size1 + size2;
    }

    template <class T1, class T2, class TPacker1, class TPacker2>
    inline size_t TPairPacker<T1, T2, TPacker1, TPacker2>::SkipLeaf(const char* buffer) const {
        size_t size1 = TPacker1().SkipLeaf(buffer);
        size_t size2 = TPacker2().SkipLeaf(buffer + size1);
        return size1 + size2;
    }

    //------------------------------------------------------------------------------------------
    // Packer for fixed-size arrays, i.e. for std::array.
    // Saves memory by not storing anything about their size.
    // SkipLeaf skips every value, so can be slow for big arrays.
    // Requires std::tuple_size<TValue>, TValue::operator[] and possibly TValue::value_type.
    template <class TValue, class TElementPacker = TPacker<typename TValue::value_type>>
    class TArrayPacker {
    public:
        using TElemPacker = TElementPacker;

        enum {
            Size = std::tuple_size<TValue>::value
        };

        void UnpackLeaf(const char* p, TValue& t) const {
            const char* buf = p;
            for (size_t i = 0; i < Size; ++i) {
                TElemPacker().UnpackLeaf(buf, t[i]);
                buf += TElemPacker().SkipLeaf(buf);
            }
        }

        void PackLeaf(char* buffer, const TValue& data, size_t computedSize) const {
            size_t remainingSize = computedSize;
            char* pos = buffer;
            for (size_t i = 0; i < Size; ++i) {
                const size_t elemSize = TElemPacker().MeasureLeaf(data[i]);
                TElemPacker().PackLeaf(pos, data[i], Min(elemSize, remainingSize));
                pos += elemSize;
                remainingSize -= elemSize;
            }
        }

        size_t MeasureLeaf(const TValue& data) const {
            size_t result = 0;
            for (size_t i = 0; i < Size; ++i) {
                result += TElemPacker().MeasureLeaf(data[i]);
            }
            return result;
        }

        size_t SkipLeaf(const char* p) const // this function better be fast because it is very frequently used
        {
            const char* buf = p;
            for (size_t i = 0; i < Size; ++i) {
                buf += TElemPacker().SkipLeaf(buf);
            }
            return buf - p;
        }
    };

    //------------------------------------
    // TPacker --- the generic packer.

    template <class T, bool IsIntegral>
    class TPackerImpl;

    template <class T>
    class TPackerImpl<T, true>: public TIntegralPacker<T> {
    };
    // No implementation for non-integral types.

    template <class T>
    class TPacker: public TPackerImpl<T, std::is_integral<T>::value> {
    };

    template <>
    class TPacker<float>: public TAsIsPacker<float> {
    };

    template <>
    class TPacker<double>: public TAsIsPacker<double> {
    };

    template <>
    class TPacker<TString>: public TStringPacker<TString> {
    };

    template <>
    class TPacker<TUtf16String>: public TStringPacker<TUtf16String> {
    };

    template <>
    class TPacker<TStringBuf>: public TStringPacker<TStringBuf> {
    };

    template <>
    class TPacker<TWtringBuf>: public TStringPacker<TWtringBuf> {
    };

    template <class T>
    class TPacker<std::vector<T>>: public TContainerPacker<std::vector<T>> {
    };

    template <class T>
    class TPacker<TVector<T>>: public TContainerPacker<TVector<T>> {
    };

    template <class T>
    class TPacker<std::list<T>>: public TContainerPacker<std::list<T>> {
    };

    template <class T>
    class TPacker<TList<T>>: public TContainerPacker<TList<T>> {
    };

    template <class T>
    class TPacker<std::set<T>>: public TContainerPacker<std::set<T>> {
    };

    template <class T>
    class TPacker<TSet<T>>: public TContainerPacker<TSet<T>> {
    };

    template <class T1, class T2>
    class TPacker<std::pair<T1, T2>>: public TPairPacker<T1, T2> {
    };

    template <class T, size_t N>
    class TPacker<std::array<T, N>>: public TArrayPacker<std::array<T, N>> {
    };

}
