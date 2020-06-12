
#include <catboost/libs/data/unaligned_mem.h>

#include <catboost/libs/helpers/vector_helpers.h>

#include <util/system/types.h>

#include <cstring>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(TUnalignedArrayBuf) {
    Y_UNIT_TEST(Basic) {
        for (auto shift : {0, 1, 2, 3}) {
            for (auto createFromArrayBuf : {false, true}) {
                TVector<float> data = {0.0f, 0.12f, 0.22f};
                TVector<ui8> unalignedData(sizeof(float)*3 + shift);
                memcpy(unalignedData.data() + shift, data.data(), sizeof(float)*3);

                TUnalignedArrayBuf<float> unalignedArrayBuf = createFromArrayBuf ?
                        TUnalignedArrayBuf<float>(
                            TConstArrayRef<ui8>(unalignedData.data() + shift, sizeof(float)*3)
                        )
                        : TUnalignedArrayBuf<float>(unalignedData.data() + shift, sizeof(float)*3);

                TVector<float> dataCopy;

                unalignedArrayBuf.WriteTo(&dataCopy);

                UNIT_ASSERT_VALUES_EQUAL(data, dataCopy);

                size_t i = 0;
                for (auto it = unalignedArrayBuf.GetIterator(); !it.AtEnd(); it.Next(), ++i) {
                    UNIT_ASSERT(i < 3);
                    UNIT_ASSERT_VALUES_EQUAL(data[i], it.Cur());
                }
                UNIT_ASSERT(i == 3);

                TMaybeOwningArrayHolder<float> alignedData = unalignedArrayBuf.GetAlignedData();
                UNIT_ASSERT(Equal(TConstArrayRef<float>(*alignedData), data));
            }
        }
    }
}
