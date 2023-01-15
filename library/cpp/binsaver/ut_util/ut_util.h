#pragma once

#include <library/cpp/binsaver/bin_saver.h>
#include <library/cpp/binsaver/mem_io.h>
#include <library/cpp/binsaver/util_stream_io.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/vector.h>
#include <util/stream/buffer.h>

#include <functional>


/* comparerChecksInside == true means comparer uses UNIT_ASSERT... inside itself
 * comparerChecksInside == false means comparer returns if its arguments are equal
 */

template <class T, class TComparer = std::equal_to<T>, bool comparerChecksInside = false>
void UnitTestCheckWithComparer(const T& lhs, const T& rhs, const TComparer& comparer) {
    if constexpr (comparerChecksInside) {
        comparer(lhs, rhs);
    } else {
        UNIT_ASSERT(comparer(lhs, rhs));
    }
}


/* comparerChecksInside == true means comparer uses UNIT_ASSERT... inside itself
 * comparerChecksInside == false means comparer returns true if its arguments are equal
 */

template <typename T, typename TComparer = std::equal_to<T>, bool comparerChecksInside = false>
void TestBinSaverSerializationToBuffer(const T& original, const TComparer& comparer = TComparer()) {
    TBufferOutput out;
    {
        TYaStreamOutput yaOut(out);

        IBinSaver f(yaOut, false, false);
        f.Add(0, const_cast<T*>(&original));
    }
    TBufferInput in(out.Buffer());
    T restored;
    {
        TYaStreamInput yaIn(in);
        IBinSaver f(yaIn, true, false);
        f.Add(0, &restored);
    }
    UnitTestCheckWithComparer<T, TComparer, comparerChecksInside>(original, restored, comparer);
}

template <typename T, typename TComparer = std::equal_to<T>, bool comparerChecksInside = false>
void TestBinSaverSerializationToVector(const T& original, const TComparer& comparer = TComparer()) {
    TVector<char> out;
    SerializeToMem(&out, *const_cast<T*>(&original));
    T restored;
    SerializeFromMem(&out, restored);
    UnitTestCheckWithComparer<T, TComparer, comparerChecksInside>(original, restored, comparer);

    TVector<TVector<char>> out2D;
    SerializeToMem(&out2D, *const_cast<T*>(&original));
    T restored2D;
    SerializeFromMem(&out2D, restored2D);
    UnitTestCheckWithComparer<T, TComparer, comparerChecksInside>(original, restored2D, comparer);
}

template <typename T, typename TComparer = std::equal_to<T>, bool comparerChecksInside = false>
void TestBinSaverSerialization(const T& original, const TComparer& comparer = TComparer()) {
    TestBinSaverSerializationToBuffer<T, TComparer, comparerChecksInside>(original, comparer);
    TestBinSaverSerializationToVector<T, TComparer, comparerChecksInside>(original, comparer);
}
