#include <catboost/libs/helpers/sparse_array.h>

#include <catboost/libs/helpers/vector_helpers.h>

#include <library/cpp/binsaver/ut_util/ut_util.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/random/shuffle.h>
#include <util/system/compiler.h>

#include <cmath>
#include <type_traits>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(SparseArray) {
    Y_UNIT_TEST(CheckIsIncreasingIndicesArray) {
        auto checkGood = [&] (const TVector<ui32>& array) {
            CheckIsIncreasingIndicesArray<ui32>(array, "array", false);
        };

        checkGood({});
        checkGood({0});
        checkGood({12});
        checkGood({0, 5, 6, 12, 100});
        checkGood({5, 7, 8, 9});

        auto checkBad = [&] (const TVector<int>& array) {
            UNIT_ASSERT_EXCEPTION(
                CheckIsIncreasingIndicesArray<int>(array, "array", false),
                TCatBoostException);
        };

        checkBad({-1});
        checkBad({-2, 0, 7});
        checkBad({0, -2, 7});
        checkBad({0, 0, 0});
        checkBad({0, 7, 8, 8, 12});
        checkBad({9, 10, 11, 7, 14});
    }

    Y_UNIT_TEST(TSparseSubsetIndices) {
        auto checkGood = [&] (TVector<int>&& array, int expectedSize, int expectedUpperBound) {
            TSparseSubsetIndices<int> ssi(std::move(array));
            ssi.Check();
            UNIT_ASSERT_VALUES_EQUAL(ssi.GetSize(), expectedSize);
            UNIT_ASSERT_VALUES_EQUAL(ssi.GetUpperBound(), expectedUpperBound);
        };

        checkGood({}, 0, 0);
        checkGood({0}, 1, 1);
        checkGood({12}, 1, 13);
        checkGood({0, 5, 6, 12, 100}, 5, 101);
        checkGood({5, 7, 8, 9}, 4, 10);

        auto checkBad = [&] (TVector<int>&& array) {
            TSparseSubsetIndices<int> ssi(std::move(array));
            UNIT_ASSERT_EXCEPTION(ssi.Check(), TCatBoostException);
        };

        checkBad({-1});
        checkBad({-2, 0, 7});
        checkBad({0, -2, 7});
        checkBad({0, 0, 0});
        checkBad({0, 7, 8, 8, 12});
        checkBad({9, 10, 11, 7, 14});
    }

    Y_UNIT_TEST(TSparseSubsetBlocks) {
        auto checkGood = [&] (
            TVector<int>&& blockStarts,
            TVector<int>&& blockLengths,
            int expectedSize,
            int expectedUpperBound)
        {
            TSparseSubsetBlocks<int> ssb(std::move(blockStarts), std::move(blockLengths));
            ssb.Check();
            UNIT_ASSERT_VALUES_EQUAL(ssb.GetSize(), expectedSize);
            UNIT_ASSERT_VALUES_EQUAL(ssb.GetUpperBound(), expectedUpperBound);
        };

        checkGood({}, {}, 0, 0);
        checkGood({0}, {0}, 0, 0);
        checkGood({12}, {0}, 0, 12);
        checkGood({0, 1}, {0, 0}, 0, 1);
        checkGood({0, 1}, {0, 1}, 1, 2);
        checkGood({0, 1, 12}, {1, 0, 2}, 3, 14);
        checkGood({5, 7, 11}, {2, 1, 9}, 12, 20);


        auto checkBad = [&] (TVector<int>&& blockStarts, TVector<int>&& blockLengths) {
            TSparseSubsetBlocks<int> ssb(std::move(blockStarts), std::move(blockLengths));
            UNIT_ASSERT_EXCEPTION(ssb.Check(), TCatBoostException);
        };

        checkBad({-1}, {0});
        checkBad({0}, {-2});
        checkBad({0, -1}, {0, 0});
        checkBad({0, 0}, {1, 0});
        checkBad({3, 3}, {2, 1});
        checkBad({1, 3, 8}, {2, 1, -3});
        checkBad({1, 3, 8}, {2, 10, 2});
        checkBad({3, 1, 8}, {2, 0, 2});
        checkBad({1, 3}, {2, 1, 2});
        checkBad({0, 2, 4}, {});
    }

    Y_UNIT_TEST(TSparseSubsetBlocksIterator) {
        auto checkCase = [] (
            TVector<ui32>&& blockStarts,
            TVector<ui32>&& blockLengths,
            const TVector<ui32>& expectedIndices) {

            TSparseSubsetBlocks<ui32> sparseSubsetBlocks(std::move(blockStarts), std::move(blockLengths));

            UNIT_ASSERT(
                (AreSequencesEqual<ui32>(
                    MakeHolder<TStaticIteratorRangeAsDynamic<const ui32*>>(expectedIndices),
                    MakeHolder<TSparseSubsetBlocksIterator<ui32>>(sparseSubsetBlocks))));
        };

        checkCase({}, {}, {});

        checkCase(/*blockStarts*/ {0, 3, 7}, /*blockLengths*/ {1, 0, 2}, /*expectedIndices*/ {0, 7, 8});

        checkCase(
            /*blockStarts*/ {3, 64 * 2 + 3, 64 * 7, 64 * 7 + 11, 64 * 7 + 14},
            /*blockLengths*/ {2, 3, 1, 2, 1},
            /*expectedIndices*/ {
                3,
                4,
                64 * 2 + 3,
                64 * 2 + 4,
                64 * 2 + 5,
                64 * 7,
                64 * 7 + 11,
                64 * 7 + 12,
                64 * 7 + 14
            });
    }

    Y_UNIT_TEST(TSparseSubsetHybridIndex) {
        auto checkGood = [&] (
            TVector<int>&& blockIndices,
            TVector<ui64>&& blockBitmaps,
            int expectedSize,
            int expectedUpperBound)
        {
            TSparseSubsetHybridIndex<int> sshi{std::move(blockIndices), std::move(blockBitmaps)};
            sshi.Check();
            UNIT_ASSERT_VALUES_EQUAL(sshi.GetSize(), expectedSize);
            UNIT_ASSERT_VALUES_EQUAL(sshi.GetUpperBound(), expectedUpperBound);
        };

        checkGood({}, {}, 0, 0);
        checkGood({0}, {0}, 0, 0);
        checkGood({0}, {0b1}, 1, 1);
        checkGood({0}, {0b1001}, 2, 4);
        checkGood({0, 3}, {0, 0b10111}, 4, 64 * 3 + 5);
        checkGood({0, 3, 4}, {0, 0b10111, 0}, 4, 64 * 4);
        checkGood({4, 7, 12}, {0b10, 0, 0b11111}, 6, 64 * 12 + 5);
        checkGood({4, 6, 13}, {0b101, 0b1001, 0b11111}, 9, 64 * 13 + 5);

        auto checkBad = [&] (TVector<int>&& blockIndices, TVector<ui64>&& blockBitmaps) {
            TSparseSubsetHybridIndex<int> sshi{std::move(blockIndices), std::move(blockBitmaps)};
            UNIT_ASSERT_EXCEPTION(sshi.Check(), TCatBoostException);
        };

        checkBad({0}, {});
        checkBad({}, {0});
        checkBad({-1}, {0});
        checkBad({0, -1, 2}, {0, 0b1, 0b110});
        checkBad({0, 0, 2}, {0, 0b1, 0b110});
        checkBad({0, 3, 3}, {0, 0b1, 0b110});
        checkBad({7, 1, 10}, {0, 0b1, 0b110});
    }

    Y_UNIT_TEST(TSparseSubsetHybridIndexIterator) {
       auto checkCase = [] (
           TVector<ui32>&& blockIndices,
           TVector<ui64>&& blockBitmaps,
           const TVector<ui32>& expectedIndices) {

           TSparseSubsetHybridIndex<ui32> sparseSubsetHybridIndex{
               std::move(blockIndices),
               std::move(blockBitmaps)
           };

           UNIT_ASSERT(
               (AreSequencesEqual<ui32>(
                   MakeHolder<TStaticIteratorRangeAsDynamic<const ui32*>>(expectedIndices),
                   MakeHolder<TSparseSubsetHybridIndexIterator<ui32>>(sparseSubsetHybridIndex))));
       };

       checkCase({}, {}, {});

       checkCase(
           /*blockIndices*/ {0, 2, 7},
           /*blockBitmaps*/ {0b11000, 0b111000, 0b101100000000001},
           /*expectedIndices*/ {
               3,
               4,
               64 * 2 + 3,
               64 * 2 + 4,
               64 * 2 + 5,
               64 * 7,
               64 * 7 + 11,
               64 * 7 + 12,
               64 * 7 + 14
           });
   }

    Y_UNIT_TEST(TSparseArrayIndexing) {
        auto checkGood = [] (
            auto&& indicesArg,
            TMaybe<int> sizeArg,
            int expectedNonDefaultSize,
            int expectedSize,
            ESparseArrayIndexingType expectedType) {

            TSparseArrayIndexing<int> sai(std::move(indicesArg), sizeArg);

            UNIT_ASSERT_VALUES_EQUAL(sai.GetNonDefaultSize(), expectedNonDefaultSize);
            UNIT_ASSERT_VALUES_EQUAL(sai.GetSize(), expectedSize);
            UNIT_ASSERT_VALUES_EQUAL(sai.GetType(), expectedType);
        };

        {
            for (auto sizeArg : {TMaybe<int>(10), TMaybe<int>()}) {
                TVector<int> indices = {};
                TSparseSubsetIndices<int> ssi(std::move(indices));
                checkGood(
                    std::move(ssi),
                    sizeArg,
                    0,
                    sizeArg ? *sizeArg : 0,
                    ESparseArrayIndexingType::Indices);
            }
        }
        {
            for (auto sizeArg : {TMaybe<int>(20), TMaybe<int>()}) {
                TVector<int> indices = {5, 8, 12};
                TSparseSubsetIndices<int> ssi(std::move(indices));
                checkGood(
                    std::move(ssi),
                    sizeArg,
                    3,
                    sizeArg ? *sizeArg : 13,
                    ESparseArrayIndexingType::Indices);
            }
        }

        {
            for (auto sizeArg : {TMaybe<int>(10), TMaybe<int>()}) {
                TVector<int> blockStarts = {};
                TVector<int> blockLengths = {};
                checkGood(
                    TSparseSubsetBlocks<int>(std::move(blockStarts), std::move(blockLengths)),
                    sizeArg,
                    0,
                    sizeArg ? *sizeArg : 0,
                    ESparseArrayIndexingType::Blocks);
            }
        }
        {
            for (auto sizeArg : {TMaybe<int>(20), TMaybe<int>()}) {
                TVector<int> blockStarts = {2, 4, 7};
                TVector<int> blockLengths = {2, 1, 11};
                checkGood(
                    TSparseSubsetBlocks<int>(std::move(blockStarts), std::move(blockLengths)),
                    sizeArg,
                    14,
                    sizeArg ? *sizeArg : 18,
                    ESparseArrayIndexingType::Blocks);
            }
        }

        {
            for (auto sizeArg : {TMaybe<int>(10), TMaybe<int>()}) {
                checkGood(
                    TSparseSubsetHybridIndex<int>{{}, {}},
                    sizeArg,
                    0,
                    sizeArg ? *sizeArg : 0,
                    ESparseArrayIndexingType::HybridIndex);
            }
        }
        {
            for (auto sizeArg : {TMaybe<int>(1000), TMaybe<int>()}) {
                checkGood(
                    TSparseSubsetHybridIndex<int>{{0, 2, 9}, {0b11, 0b1, 0b10111}},
                    sizeArg,
                    7,
                    sizeArg ? *sizeArg : (9*64 + 5),
                    ESparseArrayIndexingType::HybridIndex);
            }
        }
    }




    template <class T>
    void TestEqualTo(
        /* elements that differ only by inner index are non-strictly equal
         * elements that differ by outer index are not equal (both strictly and non-strictly)
         */
        const TVector<TVector<T>>& testData) {

        for (auto outerIdx1 : xrange(testData.size())) {
            for (auto outerIdx2 : xrange(testData.size())) {
                for (auto innerIdx1 : xrange(testData[outerIdx1].size())) {
                    for (auto innerIdx2 : xrange(testData[outerIdx2].size())) {
                        if (outerIdx1 == outerIdx2) {
                            UNIT_ASSERT(
                                testData[outerIdx1][innerIdx1].EqualTo(testData[outerIdx1][innerIdx2], true)
                                    == (innerIdx1 == innerIdx2)
                            );

                            UNIT_ASSERT(
                                testData[outerIdx1][innerIdx1].EqualTo(testData[outerIdx1][innerIdx2], false)
                            );
                        } else {
                            for (auto strict : {false, true}) {
                                UNIT_ASSERT(
                                    !testData[outerIdx1][innerIdx1].EqualTo(
                                        testData[outerIdx2][innerIdx2],
                                        strict
                                    )
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /*
     * In result:
     * elements that differ only by inner index are non-strictly equal
     * elements that differ by outer index are not equal (both strictly and non-strictly)
     */
    TVector<TVector<TSparseArrayIndexing<ui32>>> GenerateSparseArrayIndexingTestDataForEqualTo() {
        TVector<TVector<TSparseArrayIndexing<ui32>>> testData;

        testData.push_back(
            TVector<TSparseArrayIndexing<ui32>>{
                TSparseArrayIndexing<ui32>(
                    TSparseSubsetIndices<ui32>(
                        TVector<ui32>{0, 1, 64 * 2, 64 * 9, 64 * 9 + 1, 64 * 9 + 2, 64 * 9 + 4}
                    )
                ),
                TSparseArrayIndexing<ui32>(
                    TSparseSubsetBlocks<ui32>(
                        TVector<ui32>{0, 64 * 2, 64 * 9, 64 * 9 + 4},
                        TVector<ui32>{2, 1, 3, 1}
                    )
                ),
                TSparseArrayIndexing<ui32>(TSparseSubsetHybridIndex<ui32>{{0, 2, 9}, {0b11, 0b1, 0b10111}})
            }
        );

        testData.push_back(
            TVector<TSparseArrayIndexing<ui32>>{
                TSparseArrayIndexing<ui32>(
                    TSparseSubsetIndices<ui32>(
                        TVector<ui32>{
                            3,
                            4,
                            64 * 2 + 3,
                            64 * 2 + 4,
                            64 * 2 + 5,
                            64 * 7,
                            64 * 7 + 11,
                            64 * 7 + 12,
                            64 * 7 + 14
                        }
                    )
                ),
                TSparseArrayIndexing<ui32>(
                    TSparseSubsetBlocks<ui32>(
                        TVector<ui32>{3, 64 * 2 + 3, 64 * 7, 64 * 7 + 11, 64 * 7 + 14},
                        TVector<ui32>{2, 3, 1, 2, 1}
                    )
                ),
                TSparseArrayIndexing<ui32>(
                    TSparseSubsetHybridIndex<ui32>{{0, 2, 7}, {0b11000, 0b111000, 0b101100000000001}}
                )
            }
        );
        return testData;
    }


    Y_UNIT_TEST(TSparseArrayIndexingEquality) {
        TestEqualTo(GenerateSparseArrayIndexingTestDataForEqualTo());
    }

    Y_UNIT_TEST(TSparseArrayIndexingBinSaverSerialization) {
        TestBinSaverSerialization(
            TSparseArrayIndexing<ui32>(TSparseSubsetIndices<ui32>(TVector<ui32>{1, 3, 12}))
        );
        TestBinSaverSerialization(
            TSparseArrayIndexing<ui32>(
                TSparseSubsetBlocks<ui32>(TVector<ui32>{1, 3, 9}, TVector<ui32>{0, 4, 2})
            )
        );
        TestBinSaverSerialization(
            TSparseArrayIndexing<ui32>(TSparseSubsetHybridIndex<ui32>{{0, 2, 9}, {0b11, 0b1, 0b10111}})
        );
    }

    Y_UNIT_TEST(TSparseArrayIndexingBuilder) {
        auto addIndicesToBuilder = [&] (auto& builder, TVector<ui32>&& indices, bool addOrdered) {
            if (!addOrdered) {
                Shuffle(indices.begin(), indices.end(), TReallyFastRng32(0));
            }

            for (auto i : indices) {
                if (addOrdered) {
                    builder.AddOrdered(i);
                } else {
                    builder.AddNonOrdered(i);
                }
            }
        };

        {
            for (auto addOrdered : {true, false}) {
                TSparseSubsetIndicesBuilder<ui32> builder;
                addIndicesToBuilder(builder, {3, 12, 14, 15}, addOrdered);

                TSparseArrayIndexing<ui32> result = builder.Build();
                const TSparseSubsetIndices<ui32>& sparseSubsetIndices
                   = std::get<TSparseSubsetIndices<ui32>>(result.GetImpl());

                UNIT_ASSERT(Equal(*sparseSubsetIndices, TVector<ui32>{3, 12, 14, 15}));
            }
        }
        {
            for (auto addOrdered : {true, false}) {
                TSparseSubsetBlocksBuilder<ui32> builder;

                addIndicesToBuilder(builder, {3, 4, 12, 14, 15, 16}, addOrdered);

                TSparseArrayIndexing<ui32> result = builder.Build();
                const TSparseSubsetBlocks<ui32>& sparseSubsetBlocks
                    = std::get<TSparseSubsetBlocks<ui32>>(result.GetImpl());

                UNIT_ASSERT(Equal(*sparseSubsetBlocks.BlockStarts, TVector<ui32>{3, 12, 14}));
                UNIT_ASSERT(Equal(*sparseSubsetBlocks.BlockLengths, TVector<ui32>{2, 1, 3}));
            }
        }
        {
            for (auto addOrdered : {true, false}) {
                TSparseSubsetHybridIndexBuilder<ui32> builder;

                addIndicesToBuilder(
                    builder,
                    {3, 4, 64 * 2 + 3, 64 * 2 + 4, 64 * 2 + 5, 64 * 7, 64 * 7 + 11, 64 * 7 + 12, 64 * 7 + 14},
                    addOrdered);

                TSparseArrayIndexing<ui32> result = builder.Build();
                const TSparseSubsetHybridIndex<ui32>& sparseSubsetHybridIndex
                    = std::get<TSparseSubsetHybridIndex<ui32>>(result.GetImpl());

                UNIT_ASSERT_VALUES_EQUAL(sparseSubsetHybridIndex.BlockIndices, (TVector<ui32>{0, 2, 7}));
                UNIT_ASSERT_VALUES_EQUAL(
                    sparseSubsetHybridIndex.BlockBitmaps,
                    (TVector<ui64>{0b11000, 0b111000, 0b101100000000001}));
            }
        }
    }


    template <class T, class TIndexingArg>
    void TestSparseArrayBinSaverSerialization(
        TIndexingArg&& indexingArg,
        TVector<T>&& nonDefaultValues,
        T defaultValue
    ) {
        TSparseArray<T, ui32> data(
            std::move(indexingArg),
            TMaybeOwningArrayHolder<T>::CreateOwning(std::move(nonDefaultValues)),
            std::move(defaultValue));

        TestBinSaverSerialization(data);
    }

    Y_UNIT_TEST(TSparseArrayBinSaverSerialization) {
        TestSparseArrayBinSaverSerialization(
            TSparseSubsetIndices<ui32>(TVector<ui32>{1, 3, 12}),
            {0.1f, 0.2f, 1.0f},
            0.0f);
        TestSparseArrayBinSaverSerialization(
            TSparseSubsetBlocks<ui32>(TVector<ui32>{1, 3, 9}, TVector<ui32>{0, 4, 2}),
            {3.1f, 2.2f, 1.0f, 7.3f, 8.8f, 6.12f},
            0.0f);
        TestSparseArrayBinSaverSerialization(
            TSparseSubsetHybridIndex<ui32>{{0, 2, 9}, {0b11, 0b1, 0b10111}},
            {3.9f, 1.0f, 2.0f, 7.1f, 3.3f, 0.8f, 4.5f},
            0.0f);
    }

    Y_UNIT_TEST(TSparseArrayEquality) {
        TVector<TVector<TSparseArrayIndexing<ui32>>> sparseArrayIndexingTestData
            = GenerateSparseArrayIndexingTestDataForEqualTo();

        /* elements that differ only by inner index are non-strictly equal
         * elements that differ by outer index are not equal (both strictly and non-strictly)
         */
        TVector<TVector<TSparseArray<float, ui32>>> testData(6);

        auto addToTestData = [&] (
            size_t dstFirstIdx,
            size_t indexingOuterIdx,
            size_t indexingInnerIdx,
            const TVector<float>& nonDefaultValues,
            float defaultValue) {

            testData[dstFirstIdx].push_back(
                TSparseArray<float, ui32>(
                    MakeIntrusive<TSparseArrayIndexing<ui32>>(
                        TSparseArrayIndexing<ui32>(
                            sparseArrayIndexingTestData[indexingOuterIdx][indexingInnerIdx])),
                    TMaybeOwningArrayHolder<float>::CreateOwning(TVector<float>(nonDefaultValues)),
                    std::move(defaultValue)));
        };

        for (auto i : xrange(3)) {
            {
                TVector<float> nonDefaultData = {3.9f, 1.0f, 2.0f, 7.1f, 3.3f, 0.8f, 4.5f};
                addToTestData(0, 0, i, nonDefaultData, 0.0f);
                addToTestData(1, 0, i, nonDefaultData, 0.1f);
            }
            {
                TVector<float> nonDefaultData = {3.9f, 1.0f, 2.0f, 7.1f, 3.1f, 0.8f, 4.5f};
                addToTestData(2, 0, i, nonDefaultData, 0.0f);
            }
            {
                TVector<float> nonDefaultData = {9.0f, 0.1f, 0.7f, 4.2f, 0.88f, 0.2f, 3.0f, 2.11f, 0.32f};
                addToTestData(3, 1, i, nonDefaultData, 0.0f);
                addToTestData(4, 1, i, nonDefaultData, 0.1f);
            }
            {
                TVector<float> nonDefaultData = {9.0f, 0.1f, 0.7f, 4.2f, 0.88f, 0.7f, 3.0f, 2.11f, 0.32f};
                addToTestData(5, 1, i, nonDefaultData, 0.0f);
            }
        }

        TestEqualTo(testData);
    }

    template <class TValue, class TIndexing, class TContainer>
    void CheckSparseArrayBaseIteration(
        TIndexing&& indexing,
        TContainer&& nonDefaultValues,
        TValue defaultValue,
        const TVector<TValue>& expectedArray,
        const TVector<ui32>& expectedNonDefaultIndicesArray,
        const TVector<TValue>& expectedNonDefaultValuesArray,
        std::function<bool(TValue, TValue)>&& areValuesEqual
            = [](TValue lhs, TValue rhs) { return lhs == rhs; }) {

        TSparseArrayBase<TValue, TContainer, ui32> sparseArray(
            MakeIntrusive<TSparseArrayIndexing<ui32>>(std::move(indexing)),
            std::move(nonDefaultValues),
            std::move(defaultValue));

        {
            size_t i = 0;
            sparseArray.ForEachNonDefault(
                [&] (ui32 nonDefaultIdx, TValue v) {
                    UNIT_ASSERT_VALUES_EQUAL(nonDefaultIdx, expectedNonDefaultIndicesArray[i]);
                    UNIT_ASSERT(areValuesEqual(v, expectedNonDefaultValuesArray[i]));
                    ++i;
                });
            UNIT_ASSERT_VALUES_EQUAL(i, expectedNonDefaultValuesArray.size());
        }

        {
            for (size_t maxBlockSize : {1, 2, 5, 10, 100, 1000}) {
                for (auto offset : xrange(expectedArray.size())) {
                    size_t i = offset;
                    auto blockIterator = sparseArray.GetBlockIterator(offset);
                    while (auto block = blockIterator->Next(maxBlockSize)) {
                        UNIT_ASSERT(block.size() <= maxBlockSize);
                        for (auto v : block) {
                            UNIT_ASSERT(areValuesEqual(v, expectedArray[i]));
                            ++i;
                        }
                    }
                    UNIT_ASSERT_VALUES_EQUAL(i, expectedArray.size());
                }
            }
        }
    };

    Y_UNIT_TEST(TSparseArrayIteration) {
        auto checkIteration = [] (
            auto&& indexing,
            TVector<float>&& nonDefaultValues,
            float defaultValue,
            const TVector<float>& expectedArray,
            const TVector<ui32>& expectedNonDefaultIndicesArray
        ) {
            CheckSparseArrayBaseIteration(
                std::move(indexing),
                TMaybeOwningArrayHolder<float>::CreateOwning(TVector<float>(nonDefaultValues)),
                defaultValue,
                expectedArray,
                expectedNonDefaultIndicesArray,
                nonDefaultValues);
        };

        {
            TVector<ui32> indices = {1, 3, 12};
            checkIteration(
                TSparseSubsetIndices<ui32>(TVector<ui32>(indices)),
                /*nonDefaultValues*/ {0.1f, 0.2f, 1.0f},
                0.0f,
                /*expectedArray*/ {
                    0.0f,
                    0.1f,
                    0.0f,
                    0.2f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f},
                /*expectedNonDefaultIndices*/ indices);
        }
        {
            TVector<ui32> indices = {0, 1, 6};
            checkIteration(
                TSparseArrayIndexing<ui32>(
                    TSparseSubsetIndices<ui32>(TVector<ui32>(indices)),
                    10),
                /*nonDefaultValues*/ {3.1f, 2.2f, 1.0f},
                0.0f,
                /*expectedArray*/ {3.1f, 2.2f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f},
                /*expectedNonDefaultIndices*/ indices);
        }

        {
            TVector<ui32> blockStarts = {1, 3, 9};
            TVector<ui32> blockLengths = {0, 4, 2};

            checkIteration(
                TSparseSubsetBlocks<ui32>(std::move(blockStarts), std::move(blockLengths)),
                /*nonDefaultValues*/ {3.1f, 2.2f, 1.0f, 7.3f, 8.8f, 6.12f},
                0.0f,
                /*expectedArray*/ {0.0f, 0.0f, 0.0f, 3.1f, 2.2f, 1.0f, 7.3f, 0.0f, 0.0f, 8.8f, 6.12f},
                /*expectedNonDefaultIndices*/ {3, 4, 5, 6, 9, 10});
        }
        {
            TVector<ui32> blockStarts = {0, 3, 7};
            TVector<ui32> blockLengths = {1, 0, 2};

            checkIteration(
                TSparseArrayIndexing<ui32>(
                    TSparseSubsetBlocks<ui32>(std::move(blockStarts), std::move(blockLengths)),
                    11),
                /*nonDefaultValues*/ {3.1f, 2.2f, 1.0f},
                0.0f,
                /*expectedArray*/ {3.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.2f, 1.0f, 0.0f, 0.0f},
                /*expectedNonDefaultIndices*/ {0, 7, 8});
        }

        {
            TVector<float> expectedValues(64 * 9 + 5, 0.0f);
            TVector<ui32> expectedNonDefaultIndices;
            expectedValues[0] = 3.9f;
            expectedNonDefaultIndices.push_back(0);
            expectedValues[1] = 1.0f;
            expectedNonDefaultIndices.push_back(1);
            expectedValues[64 * 2] = 2.0f;
            expectedNonDefaultIndices.push_back(64 * 2);
            expectedValues[64 * 9] = 7.1f;
            expectedNonDefaultIndices.push_back(64 * 9);
            expectedValues[64 * 9 + 1] = 3.3f;
            expectedNonDefaultIndices.push_back(64 * 9 + 1);
            expectedValues[64 * 9 + 2] = 0.8f;
            expectedNonDefaultIndices.push_back(64 * 9 + 2);
            expectedValues[64 * 9 + 4] = 4.5f;
            expectedNonDefaultIndices.push_back(64 * 9 + 4);

            checkIteration(
                TSparseSubsetHybridIndex<ui32>{{0, 2, 9}, {0b11, 0b1, 0b10111}},
                /*nonDefaultValues*/ {3.9f, 1.0f, 2.0f, 7.1f, 3.3f, 0.8f, 4.5f},
                0.0f,
                expectedValues,
                expectedNonDefaultIndices);
        }

        {
            TVector<float> expectedValues(1000, 0.0f);
            TVector<ui32> expectedNonDefaultIndices;
            expectedValues[64 * 1 + 2] = 3.9f;
            expectedNonDefaultIndices.push_back(64 * 1 + 2);
            expectedValues[64 * 1 + 3] = 1.0f;
            expectedNonDefaultIndices.push_back(64 * 1 + 3);
            expectedValues[64 * 3] = 2.0f;
            expectedNonDefaultIndices.push_back(64 * 3);
            expectedValues[64 * 3 + 2] = 7.1f;
            expectedNonDefaultIndices.push_back(64 * 3 + 2);
            expectedValues[64 * 3 + 4] = 3.3f;
            expectedNonDefaultIndices.push_back(64 * 3 + 4);

            checkIteration(
                TSparseArrayIndexing<ui32>(
                    TSparseSubsetHybridIndex<ui32> { { 1, 3 }, { 0b1100, 0b10101 } },
                    1000),
                /*nonDefaultValues*/ { 3.9f, 1.0f, 2.0f, 7.1f, 3.3f },
                0.0f,
                expectedValues,
                expectedNonDefaultIndices);
        }

    }

    template <class TIndexing, class TInterfaceValue, class TStoredValue>
    void TestConstPolymorphicValuesSparseArrayIterationCase(
        TIndexing&& indexing,
        TVector<TStoredValue>&& nonDefaultValues,
        TInterfaceValue defaultValue,
        const TVector<TInterfaceValue>& expectedArray,
        const TVector<ui32>& expectedNonDefaultIndicesArray
    ) {
        TVector<TInterfaceValue> expectedNonDefaultValues(
            nonDefaultValues.begin(),
            nonDefaultValues.end()
        );

        std::function<bool(TInterfaceValue, TInterfaceValue)> areValuesEqual
            = [](TInterfaceValue lhs, TInterfaceValue rhs) -> bool {
                return std::abs((double)lhs - (double)rhs) < 1.e13;
            };

        CheckSparseArrayBaseIteration(
            std::move(indexing),
            TTypedSequenceContainer<TInterfaceValue>(
                MakeTypeCastArrayHolderFromVector<TInterfaceValue>(nonDefaultValues)
            ),
            (const TInterfaceValue)defaultValue,
            expectedArray,
            expectedNonDefaultIndicesArray,
            expectedNonDefaultValues,
            std::move(areValuesEqual));
    }


    Y_UNIT_TEST(TConstPolymorphicValuesSparseArrayIteration) {
        {
            TVector<ui32> indices = {1, 3, 12};
            TestConstPolymorphicValuesSparseArrayIterationCase(
                TSparseSubsetIndices<ui32>(TVector<ui32>(indices)),
                /*nonDefaultValues*/ TVector<float>{0.1f, 0.2f, 1.0f},
                0.0,
                /*expectedArray*/ TVector<double>{
                    0.0,
                    0.1,
                    0.0,
                    0.2,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0},
                /*expectedNonDefaultIndices*/ indices);
        }
        {
            TVector<ui32> indices = {0, 1, 6};
            TestConstPolymorphicValuesSparseArrayIterationCase(
                TSparseArrayIndexing<ui32>(
                    TSparseSubsetIndices<ui32>(TVector<ui32>(indices)),
                    10),
                /*nonDefaultValues*/ TVector<i64>{31, 22, 10},
                ui32(0),
                /*expectedArray*/ TVector<ui32>{31, 22, 0, 0, 0, 0, 10, 0, 0, 0},
                /*expectedNonDefaultIndices*/ indices);
        }

        {
            TVector<ui32> blockStarts = {1, 3, 9};
            TVector<ui32> blockLengths = {0, 4, 2};

            TestConstPolymorphicValuesSparseArrayIterationCase(
                TSparseSubsetBlocks<ui32>(std::move(blockStarts), std::move(blockLengths)),
                /*nonDefaultValues*/ TVector<ui32>{31, 22, 10, 73, 88, 612},
                0.0f,
                /*expectedArray*/TVector<float> {
                    0.0f,
                    0.0f,
                    0.0f,
                    31.0f,
                    22.0f,
                    10.0f,
                    73.0f,
                    0.0f,
                    0.0f,
                    88.0f,
                    612.0f
                },
                /*expectedNonDefaultIndices*/ {3, 4, 5, 6, 9, 10});
        }

    }


    Y_UNIT_TEST(TSparseCompressedArrayIteration) {
        auto checkIteration = [] (
            auto&& indexing,
            TVector<ui32>&& nonDefaultValues,
            ui32 bitsPerKey,
            ui32 defaultValue,
            const TVector<ui32>& expectedArray,
            const TVector<ui32>& expectedNonDefaultIndicesArray
        ) {
            CheckSparseArrayBaseIteration(
                std::move(indexing),
                TCompressedArray(
                    nonDefaultValues.size(),
                    bitsPerKey,
                    CompressVector<ui64>(nonDefaultValues, bitsPerKey)),
                defaultValue,
                expectedArray,
                expectedNonDefaultIndicesArray,
                nonDefaultValues);
        };

        {
            TVector<ui32> indices = {1, 3, 12};
            checkIteration(
                TSparseSubsetIndices<ui32>(TVector<ui32>(indices)),
                /*nonDefaultValues*/ { 1, 0, 1 },
                /*bitsPerKey*/ 1,
                /*defaultValue*/ 0,
                /*expectedArray*/ { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
                /*expectedNonDefaultIndices*/ indices);
        }
        {
            TVector<ui32> blockStarts = {1, 3, 9};
            TVector<ui32> blockLengths = {0, 4, 2};

            checkIteration(
                TSparseSubsetBlocks<ui32>(std::move(blockStarts), std::move(blockLengths)),
                /*nonDefaultValues*/ {0b110, 0b1000, 0b10010, 0b1011011, 0b111, 0b10001000},
                /*bitsPerKey*/ 8,
                /*defaultValue*/ 0b1,
                /*expectedArray*/ {
                    0b1,
                    0b1,
                    0b1,
                    0b110,
                    0b1000,
                    0b10010,
                    0b1011011,
                    0b1,
                    0b1,
                    0b111,
                    0b10001000 },
                /*expectedNonDefaultIndices*/ {3, 4, 5, 6, 9, 10}
            );
        }
        {
            TVector<ui32> expectedValues(64 * 9 + 5, 0);
            TVector<ui32> expectedNonDefaultIndices;
            expectedValues[0] = 0xF2;
            expectedNonDefaultIndices.push_back(0);
            expectedValues[1] = 0xFFEF0122;
            expectedNonDefaultIndices.push_back(1);
            expectedValues[64 * 2] = 0x126A;
            expectedNonDefaultIndices.push_back(64 * 2);
            expectedValues[64 * 9] = 0x89763A1;
            expectedNonDefaultIndices.push_back(64 * 9);
            expectedValues[64 * 9 + 1] = 0x12ADE43;
            expectedNonDefaultIndices.push_back(64 * 9 + 1);
            expectedValues[64 * 9 + 2] = 0xBA5A48CD;
            expectedNonDefaultIndices.push_back(64 * 9 + 2);
            expectedValues[64 * 9 + 4] = 0x61985128;
            expectedNonDefaultIndices.push_back(64 * 9 + 4);

            checkIteration(
                TSparseSubsetHybridIndex<ui32>{{0, 2, 9}, {0b11, 0b1, 0b10111}},
                /*nonDefaultValues*/ {
                    0xF2,
                    0xFFEF0122,
                    0x126A,
                    0x89763A1,
                    0x12ADE43,
                    0xBA5A48CD,
                    0x61985128 },
                /*bitsPerKey*/ 32,
                /*defaultValue*/ 0,
                expectedValues,
                expectedNonDefaultIndices);
        }
    }

    Y_UNIT_TEST(TSparseArrayGetSubset) {
        auto getSubset = [](
            auto srcIndexingArg,
            TVector<float> srcNonDefaultValues,
            float defaultValue,
            auto subsetIndexingArg,
            ESparseArrayIndexingType subsetSparseIndexingType) -> TSparseArray<float, ui32> {

            TSparseArray<float, ui32> sparseArray(
                MakeIntrusive<TSparseArrayIndexing<ui32>>(std::move(srcIndexingArg)),
                TMaybeOwningArrayHolder<float>::CreateOwning(std::move(srcNonDefaultValues)),
                std::move(defaultValue));

            TArraySubsetInvertedIndexing<ui32> subsetInvertedIndexing = GetInvertedIndexing(
                TArraySubsetIndexing<ui32>(std::move(subsetIndexingArg)),
                sparseArray.GetSize(),
                &NPar::LocalExecutor()
            );

            return sparseArray.GetSubset(subsetInvertedIndexing, subsetSparseIndexingType);
        };

        auto checkNonDefaultValues = [] (
            const TSparseArray<float, ui32>& sparseArray,
            TConstArrayRef<float> expectedNonDefaultValues) {

            auto expectedValueIter = expectedNonDefaultValues.begin();
            sparseArray.ForEachNonDefault(
                [&] (ui32 i, float value) {
                    Y_UNUSED(i);
                    UNIT_ASSERT_VALUES_EQUAL(value, *expectedValueIter);
                    ++expectedValueIter;
                }
            );
            UNIT_ASSERT_EQUAL(expectedValueIter, expectedNonDefaultValues.end());
        };

        {
            auto subsetSparseArray = getSubset(
                TSparseSubsetIndices<ui32>(TVector<ui32>()),
                {},
                0.0f,
                TFullSubset<ui32>(0),
                ESparseArrayIndexingType::Undefined
            );

            auto indices = std::get<TSparseSubsetIndices<ui32>>(subsetSparseArray.GetIndexing()->GetImpl());
            UNIT_ASSERT(Equal<ui32>(*indices, TVector<ui32>()));
            checkNonDefaultValues(subsetSparseArray, TVector<float>());
        }
        {
            TVector<ui32> sparseSubsetIndices{1, 7, 12, 13};
            TVector<float> nonDefaultValues{9.0f, 0.1f, 0.7f, 4.2f};

            auto subsetSparseArray = getSubset(
                TSparseSubsetIndices<ui32>(TVector<ui32>(sparseSubsetIndices)),
                nonDefaultValues,
                0.0f,
                TFullSubset<ui32>(14),
                ESparseArrayIndexingType::Undefined
            );

            auto indices = std::get<TSparseSubsetIndices<ui32>>(subsetSparseArray.GetIndexing()->GetImpl());
            UNIT_ASSERT(Equal<ui32>(*indices, sparseSubsetIndices));
            checkNonDefaultValues(subsetSparseArray, nonDefaultValues);
        }
        {
            auto subsetSparseArray = getSubset(
                TSparseArrayIndexing<ui32>(TSparseSubsetIndices<ui32>(TVector<ui32>{1, 7, 12, 13}), 16),
                /*srcNonDefaultValues*/ {9.0f, 0.1f, 0.7f, 4.2f},
                0.0f,
                TIndexedSubset<ui32>{0, 3, 14},
                ESparseArrayIndexingType::Undefined
            );

            auto indices = std::get<TSparseSubsetIndices<ui32>>(subsetSparseArray.GetIndexing()->GetImpl());
            UNIT_ASSERT((*indices).empty());
            checkNonDefaultValues(subsetSparseArray, TVector<float>());
        }
        {
            auto subsetSparseArray = getSubset(
                TSparseArrayIndexing<ui32>(TSparseSubsetIndices<ui32>(TVector<ui32>{1, 7, 12, 13}), 25),
                /*srcNonDefaultValues*/ {9.0f, 0.1f, 0.7f, 4.2f},
                0.0f,
                TIndexedSubset<ui32>{7, 13, 22},
                ESparseArrayIndexingType::Undefined
            );

            auto indices = std::get<TSparseSubsetIndices<ui32>>(subsetSparseArray.GetIndexing()->GetImpl());
            UNIT_ASSERT(Equal<ui32>(*indices, TVector<ui32>{0, 1}));
            checkNonDefaultValues(subsetSparseArray, TVector<float>{0.1f, 4.2f});
        }
        {
            auto subsetSparseArray = getSubset(
                TSparseArrayIndexing<ui32>(TSparseSubsetIndices<ui32>(TVector<ui32>{1, 7, 12, 13, 16}), 25),
                /*srcNonDefaultValues*/ {9.0f, 0.1f, 0.7f, 4.2f, 0.88f},
                0.0f,
                TIndexedSubset<ui32>{24, 0, 8, 13, 3, 1, 16},
                ESparseArrayIndexingType::Undefined
            );

            auto indices = std::get<TSparseSubsetIndices<ui32>>(subsetSparseArray.GetIndexing()->GetImpl());
            UNIT_ASSERT(Equal<ui32>(*indices, TVector<ui32>{3, 5, 6}));
            checkNonDefaultValues(subsetSparseArray, TVector<float>{4.2f, 9.0f, 0.88f});
        }
        {
            auto subsetSparseArray = getSubset(
                TSparseArrayIndexing<ui32>(
                    TSparseSubsetIndices<ui32>(TVector<ui32>{1, 7, 12, 13, 16, 17, 18, 22, 25}),
                    26
                ),
                /*srcNonDefaultValues*/ {9.0f, 0.1f, 0.7f, 4.2f, 0.88f, 0.2f, 3.0f, 2.11f, 0.32f},
                0.0f,
                TIndexedSubset<ui32>{22, 16, 12, 0, 7, 25, 3, 6, 13},
                ESparseArrayIndexingType::Blocks
            );

            auto indexBlocks = std::get<TSparseSubsetBlocks<ui32>>(subsetSparseArray.GetIndexing()->GetImpl());
            UNIT_ASSERT(Equal<ui32>(*indexBlocks.BlockStarts, TVector<ui32>{0, 4, 8}));
            UNIT_ASSERT(Equal<ui32>(*indexBlocks.BlockLengths, TVector<ui32>{3, 2, 1}));
            checkNonDefaultValues(subsetSparseArray, TVector<float>{2.11f, 0.88f, 0.7f, 0.1f, 0.32f, 4.2f});
        }
        {
            auto subsetSparseArray = getSubset(
                TSparseArrayIndexing<ui32>(
                    TSparseSubsetIndices<ui32>(TVector<ui32>{1, 7, 12, 13, 16, 17, 18, 22, 25}),
                    26
                ),
                /*srcNonDefaultValues*/ {9.0f, 0.1f, 0.7f, 4.2f, 0.88f, 0.2f, 3.0f, 2.11f, 0.32f},
                0.0f,
                TIndexedSubset<ui32>{22, 16, 12, 0, 7, 25, 3, 6, 13},
                ESparseArrayIndexingType::Blocks
            );

            auto indexBlocks = std::get<TSparseSubsetBlocks<ui32>>(subsetSparseArray.GetIndexing()->GetImpl());
            UNIT_ASSERT(Equal<ui32>(*indexBlocks.BlockStarts, TVector<ui32>{0, 4, 8}));
            UNIT_ASSERT(Equal<ui32>(*indexBlocks.BlockLengths, TVector<ui32>{3, 2, 1}));
            checkNonDefaultValues(subsetSparseArray, TVector<float>{2.11f, 0.88f, 0.7f, 0.1f, 0.32f, 4.2f});
        }
        {
            TVector<ui32> subsetIndexing(64 * 5 + 3);

            // initialize with indices not in sparse subset by default
            Iota(subsetIndexing.begin(), subsetIndexing.end(), 64 * 12);
            Shuffle(subsetIndexing.begin(), subsetIndexing.end(), TReallyFastRng32(0));

            subsetIndexing[1] = 64 * 11 + 6;
            subsetIndexing[2] = 64 * 2 + 3;
            subsetIndexing[6] = 11;
            subsetIndexing[64 * 2 + 4] = 64 * 7 + 1;
            subsetIndexing[64 * 2 + 6] = 64 * 11 + 14;
            subsetIndexing[64 * 4 + 1] = 64 * 2 + 12;
            subsetIndexing[64 * 4 + 4] = 64 * 11 + 11;
            subsetIndexing[64 * 4 + 5] = 64 * 7;

            auto subsetSparseArray = getSubset(
                TSparseArrayIndexing<ui32>(
                    TSparseSubsetIndices<ui32>(
                        TVector<ui32>{
                            11,
                            64 * 2 + 3,
                            64 * 2 + 11,
                            64 * 2 + 12,
                            64 * 7,
                            64 * 7 + 1,
                            64 * 11 + 6,
                            64 * 11 + 11,
                            64 * 11 + 12,
                            64 * 11 + 14,
                            64 * 11 + 15
                        }
                    ),
                    64 * 20
                ),
                /*srcNonDefaultValues*/ {9.0f, 0.1f, 0.7f, 4.2f, 0.88f, 0.2f, 3.0f, 2.11f, 0.32f, 10.f, 0.63f},
                0.0f,
                subsetIndexing,
                ESparseArrayIndexingType::HybridIndex
            );

            auto hybridIndex = std::get<TSparseSubsetHybridIndex<ui32>>(subsetSparseArray.GetIndexing()->GetImpl());
            UNIT_ASSERT_VALUES_EQUAL(hybridIndex.BlockIndices, (TVector<ui32>{0, 2, 4}));
            UNIT_ASSERT_VALUES_EQUAL(
                hybridIndex.BlockBitmaps,
                (TVector<ui64>{0b1000110, 0b1010000, 0b110010})
            );
            checkNonDefaultValues(
                subsetSparseArray,
                TVector<float>{3.0f, 0.1f, 9.0f, 0.2f, 10.f, 4.2f, 2.11f, 0.88f}
            );
        }
    }

    template <class TIndexing, class TInterfaceValue, class TStoredValue, class TArraySubsetIndexingArg>
    TConstPolymorphicValuesSparseArray<TInterfaceValue, ui32>
        DoTConstPolymorphicValuesSparseArrayGetSubset(
            TIndexing&& srcIndexingArg,
            TVector<TStoredValue> srcNonDefaultValues,
            TInterfaceValue defaultValue,
            TArraySubsetIndexingArg subsetIndexingArg,
            ESparseArrayIndexingType subsetSparseIndexingType
        ) {
            TConstPolymorphicValuesSparseArray<TInterfaceValue, ui32> sparseArray
                = MakeConstPolymorphicValuesSparseArray<TInterfaceValue, TStoredValue>(
                    MakeIntrusive<TSparseArrayIndexing<ui32>>(std::move(srcIndexingArg)),
                    TMaybeOwningConstArrayHolder<TStoredValue>::CreateOwning(std::move(srcNonDefaultValues)),
                    defaultValue
                );

            TArraySubsetInvertedIndexing<ui32> subsetInvertedIndexing = GetInvertedIndexing(
                TArraySubsetIndexing<ui32>(std::move(subsetIndexingArg)),
                sparseArray.GetSize(),
                &NPar::LocalExecutor()
            );

            return sparseArray.GetSubset(subsetInvertedIndexing, subsetSparseIndexingType);
        }

    template <class TInterfaceValue, class TStoredValue>
    void CheckNonDefaultValues(
        const TConstPolymorphicValuesSparseArray<TInterfaceValue, ui32>& sparseArray,
        const TVector<TStoredValue>& expectedNonDefaultValues
    ) {
        TVector<TInterfaceValue> expectedNonDefaultValuesWithDstType(
            expectedNonDefaultValues.begin(),
            expectedNonDefaultValues.end()
        );

        auto expectedValueIter = expectedNonDefaultValuesWithDstType.begin();
        sparseArray.ForEachNonDefault(
            [&] (ui32 i, TInterfaceValue value) {
                Y_UNUSED(i);
                UNIT_ASSERT_VALUES_EQUAL(value, *expectedValueIter);
                ++expectedValueIter;
            }
        );
        UNIT_ASSERT_EQUAL(expectedValueIter, expectedNonDefaultValuesWithDstType.end());
    }


    Y_UNIT_TEST(TConstPolymorphicValuesSparseArrayGetSubset) {

        {
            TVector<ui32> sparseSubsetIndices{1, 7, 12, 13};
            TVector<ui8> nonDefaultValues{1, 255, 12, 122};

            auto subsetSparseArray = DoTConstPolymorphicValuesSparseArrayGetSubset(
                TSparseSubsetIndices<ui32>(TVector<ui32>(sparseSubsetIndices)),
                nonDefaultValues,
                ui32(0),
                TFullSubset<ui32>(14),
                ESparseArrayIndexingType::Undefined
            );

            auto indices = std::get<TSparseSubsetIndices<ui32>>(subsetSparseArray.GetIndexing()->GetImpl());
            UNIT_ASSERT(Equal<ui32>(*indices, sparseSubsetIndices));
            CheckNonDefaultValues(subsetSparseArray, nonDefaultValues);
        }
        {
            auto subsetSparseArray = DoTConstPolymorphicValuesSparseArrayGetSubset(
                TSparseArrayIndexing<ui32>(TSparseSubsetIndices<ui32>(TVector<ui32>{1, 7, 12, 13}), 16),
                TVector<ui32>{1, 255, 12, 122},
                float(0.0f),
                TIndexedSubset<ui32>{0, 3, 14},
                ESparseArrayIndexingType::Undefined
            );

            auto indices = std::get<TSparseSubsetIndices<ui32>>(subsetSparseArray.GetIndexing()->GetImpl());
            UNIT_ASSERT((*indices).empty());
            CheckNonDefaultValues(subsetSparseArray, TVector<ui32>());
        }
        {
            auto subsetSparseArray = DoTConstPolymorphicValuesSparseArrayGetSubset(
                TSparseArrayIndexing<ui32>(TSparseSubsetIndices<ui32>(TVector<ui32>{1, 7, 12, 13}), 25),
                TVector<double>{0.22, 0.17, 1.0, 0.0},
                float(0.0f),
                TIndexedSubset<ui32>{7, 13, 22},
                ESparseArrayIndexingType::Undefined
            );

            auto indices = std::get<TSparseSubsetIndices<ui32>>(subsetSparseArray.GetIndexing()->GetImpl());
            UNIT_ASSERT(Equal<ui32>(*indices, TVector<ui32>{0, 1}));
            CheckNonDefaultValues(subsetSparseArray, TVector<float>{0.17f, 0.0f});
        }
        {
            auto subsetSparseArray = DoTConstPolymorphicValuesSparseArrayGetSubset(
                TSparseArrayIndexing<ui32>(TSparseSubsetIndices<ui32>(TVector<ui32>{1, 7, 12, 13}), 25),
                TVector<float>{0.22f, 0.17f, 1.0f, 0.0f},
                float(0.0),
                TIndexedSubset<ui32>{7, 13, 22},
                ESparseArrayIndexingType::Undefined
            );

            auto indices = std::get<TSparseSubsetIndices<ui32>>(subsetSparseArray.GetIndexing()->GetImpl());
            UNIT_ASSERT(Equal<ui32>(*indices, TVector<ui32>{0, 1}));
            CheckNonDefaultValues(subsetSparseArray, TVector<float>{0.17f, 0.0f});
        }
    }

    Y_UNIT_TEST(TSparseCompressedArrayGetSubset) {
        auto getSubset = [](
            auto srcIndexingArg,
            TVector<ui8> srcNonDefaultValues,
            ui8 defaultValue,
            auto subsetIndexingArg,
            ESparseArrayIndexingType subsetSparseIndexingType) -> TSparseCompressedArray<ui8, ui32> {

            TSparseCompressedArray<ui8, ui32> sparseArray(
                MakeIntrusive<TSparseArrayIndexing<ui32>>(std::move(srcIndexingArg)),
                TCompressedArray(
                    srcNonDefaultValues.size(),
                    /*bitsPerKey*/ 8,
                    CompressVector<ui64>(srcNonDefaultValues, 8)
                ),
                std::move(defaultValue)
            );

            TArraySubsetInvertedIndexing<ui32> subsetInvertedIndexing = GetInvertedIndexing(
                TArraySubsetIndexing<ui32>(std::move(subsetIndexingArg)),
                sparseArray.GetSize(),
                &NPar::LocalExecutor()
            );

            return sparseArray.GetSubset(subsetInvertedIndexing, subsetSparseIndexingType);
        };

        auto checkNonDefaultValues = [] (
            const TSparseCompressedArray<ui8, ui32>& sparseArray,
            TConstArrayRef<ui8> expectedNonDefaultValues) {

            auto expectedValueIter = expectedNonDefaultValues.begin();
            sparseArray.ForEachNonDefault(
                [&] (ui32 i, ui8 value) {
                    Y_UNUSED(i);
                    UNIT_ASSERT_VALUES_EQUAL(value, *expectedValueIter);
                    ++expectedValueIter;
                }
            );
            UNIT_ASSERT_EQUAL(expectedValueIter, expectedNonDefaultValues.end());
        };

        {
            TVector<ui32> sparseSubsetIndices{1, 7, 12, 13};
            TVector<ui8> nonDefaultValues{0x1, 0xFF, 0x43, 0xEF};

            auto subsetSparseArray = getSubset(
                TSparseSubsetIndices<ui32>(TVector<ui32>(sparseSubsetIndices)),
                nonDefaultValues,
                0,
                TFullSubset<ui32>(14),
                ESparseArrayIndexingType::Undefined
            );

            auto indices = std::get<TSparseSubsetIndices<ui32>>(subsetSparseArray.GetIndexing()->GetImpl());
            UNIT_ASSERT(Equal<ui32>(*indices, sparseSubsetIndices));
            checkNonDefaultValues(subsetSparseArray, nonDefaultValues);
        }
        {
            auto subsetSparseArray = getSubset(
                TSparseArrayIndexing<ui32>(TSparseSubsetIndices<ui32>(TVector<ui32>{1, 7, 12, 13}), 16),
                {0x1, 0xFF, 0x43, 0xEF},
                0,
                TIndexedSubset<ui32>{0, 3, 14},
                ESparseArrayIndexingType::Undefined
            );

            auto indices = std::get<TSparseSubsetIndices<ui32>>(subsetSparseArray.GetIndexing()->GetImpl());
            UNIT_ASSERT((*indices).empty());
            checkNonDefaultValues(subsetSparseArray, TVector<ui8>());
        }
        {
            auto subsetSparseArray = getSubset(
                TSparseArrayIndexing<ui32>(TSparseSubsetIndices<ui32>(TVector<ui32>{1, 7, 12, 13}), 25),
                {0x1, 0xFF, 0x43, 0xEF},
                0,
                TIndexedSubset<ui32>{7, 13, 22},
                ESparseArrayIndexingType::Undefined
            );

            auto indices = std::get<TSparseSubsetIndices<ui32>>(subsetSparseArray.GetIndexing()->GetImpl());
            UNIT_ASSERT(Equal<ui32>(*indices, TVector<ui32>{0, 1}));
            checkNonDefaultValues(subsetSparseArray, TVector<ui8>{0xFF, 0xEF});
        }


        {
            TVector<ui32> subsetIndexing(64 * 5 + 3);

            // initialize with indices not in sparse subset by default
            Iota(subsetIndexing.begin(), subsetIndexing.end(), 64 * 12);
            Shuffle(subsetIndexing.begin(), subsetIndexing.end(), TReallyFastRng32(0));

            subsetIndexing[1] = 64 * 11 + 6;
            subsetIndexing[2] = 64 * 2 + 3;
            subsetIndexing[6] = 11;
            subsetIndexing[64 * 2 + 4] = 64 * 7 + 1;
            subsetIndexing[64 * 2 + 6] = 64 * 11 + 14;
            subsetIndexing[64 * 4 + 1] = 64 * 2 + 12;
            subsetIndexing[64 * 4 + 4] = 64 * 11 + 11;
            subsetIndexing[64 * 4 + 5] = 64 * 7;

            auto subsetSparseArray = getSubset(
                TSparseArrayIndexing<ui32>(
                    TSparseSubsetIndices<ui32>(
                        TVector<ui32>{
                            11,
                            64 * 2 + 3,
                            64 * 2 + 11,
                            64 * 2 + 12,
                            64 * 7,
                            64 * 7 + 1,
                            64 * 11 + 6,
                            64 * 11 + 11,
                            64 * 11 + 12,
                            64 * 11 + 14,
                            64 * 11 + 15
                        }
                    ),
                    64 * 20
                ),
                {0x1, 0xFF, 0x43, 0xEF, 0xAD, 0x02, 0x30, 0xA3, 0x32, 0x10, 0x08},
                0,
                subsetIndexing,
                ESparseArrayIndexingType::HybridIndex
            );

            auto hybridIndex = std::get<TSparseSubsetHybridIndex<ui32>>(subsetSparseArray.GetIndexing()->GetImpl());
            UNIT_ASSERT_VALUES_EQUAL(hybridIndex.BlockIndices, (TVector<ui32>{0, 2, 4}));
            UNIT_ASSERT_VALUES_EQUAL(
                hybridIndex.BlockBitmaps,
                (TVector<ui64>{0b1000110, 0b1010000, 0b110010})
            );
            checkNonDefaultValues(
                subsetSparseArray,
                TVector<ui8>{0x30, 0xFF, 0x1, 0x02, 0x10, 0xEF, 0xA3, 0xAD}
            );
        }
    }

}
