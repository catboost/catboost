#include <library/cpp/testing/unittest/registar.h>

#include <catboost/private/libs/data_util/line_data_reader.h>

#include <util/generic/xrange.h>
#include <util/system/tempfile.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(TBlocksSubsetLineDataReaderTest) {
    Y_UNIT_TEST(TestEmptyAll) {
        TTempFile tmpFile(MakeTempName());
        {
            TOFStream out(tmpFile.Name());
        }
        TLineDataReaderArgs lineDataReaderArgs;
        lineDataReaderArgs.PathWithScheme.Path = tmpFile.Name();
        TBlocksSubsetLineDataReader blocksSubsetLineDataReader(
            MakeHolder<TFileLineDataReader>(lineDataReaderArgs),
            TVector<TIndexRange<ui64>>()
        );
        UNIT_ASSERT(!blocksSubsetLineDataReader.GetHeader().Defined());

        UNIT_ASSERT_EQUAL(blocksSubsetLineDataReader.GetDataLineCount(), 0);

        TString line;
        UNIT_ASSERT(!blocksSubsetLineDataReader.ReadLine(&line));
    }

    Y_UNIT_TEST(TestFullSubset) {
        TTempFile tmpFile(MakeTempName());
        {
            TOFStream out(tmpFile.Name());
            out << "l0\nl1\nl2\nl3\nl4";
        }
        TLineDataReaderArgs lineDataReaderArgs;
        lineDataReaderArgs.PathWithScheme.Path = tmpFile.Name();
        TBlocksSubsetLineDataReader blocksSubsetLineDataReader(
            MakeHolder<TFileLineDataReader>(lineDataReaderArgs),
            {TIndexRange<ui64>(0, 4)}
        );
        UNIT_ASSERT(!blocksSubsetLineDataReader.GetHeader().Defined());

        UNIT_ASSERT_EQUAL(blocksSubsetLineDataReader.GetDataLineCount(), 4);

        TString line;
        for (auto expectedLine : {"l0", "l1", "l2", "l3"}) {
            UNIT_ASSERT(blocksSubsetLineDataReader.ReadLine(&line));
            UNIT_ASSERT_EQUAL(line, expectedLine);
        }
        UNIT_ASSERT(!blocksSubsetLineDataReader.ReadLine(&line));
    }

    Y_UNIT_TEST(TestEmptySubset) {
        TTempFile tmpFile(MakeTempName());
        {
            TOFStream out(tmpFile.Name());
            out << "l0\nl1\nl2\nl3\nl4";
        }
        TLineDataReaderArgs lineDataReaderArgs;
        lineDataReaderArgs.PathWithScheme.Path = tmpFile.Name();
        TBlocksSubsetLineDataReader blocksSubsetLineDataReader(
            MakeHolder<TFileLineDataReader>(lineDataReaderArgs),
            {}
        );
        UNIT_ASSERT(!blocksSubsetLineDataReader.GetHeader().Defined());

        UNIT_ASSERT_EQUAL(blocksSubsetLineDataReader.GetDataLineCount(), 0);

        TString line;
        UNIT_ASSERT(!blocksSubsetLineDataReader.ReadLine(&line));
    }

    Y_UNIT_TEST(TestNontrivialSubset) {
        TTempFile tmpFile(MakeTempName());
        {
            TOFStream out(tmpFile.Name());
            for (auto i : xrange(26)) {
                out << "l" << i << Endl;
            }
        }
        TLineDataReaderArgs lineDataReaderArgs;
        lineDataReaderArgs.PathWithScheme.Path = tmpFile.Name();
        TBlocksSubsetLineDataReader blocksSubsetLineDataReader(
            MakeHolder<TFileLineDataReader>(lineDataReaderArgs),
            {TIndexRange<ui64>(3, 11), TIndexRange<ui64>(11, 16), TIndexRange<ui64>(22, 23)}
        );
        UNIT_ASSERT(!blocksSubsetLineDataReader.GetHeader().Defined());

        UNIT_ASSERT_EQUAL(blocksSubsetLineDataReader.GetDataLineCount(), 14);

        TString line;
        auto expectedLines = TVector<TString>{
            "l3",
            "l4",
            "l5",
            "l6",
            "l7",
            "l8",
            "l9",
            "l10",
            "l11",
            "l12",
            "l13",
            "l14",
            "l15",
            "l22"
        };

        for (auto expectedLine : expectedLines) {
            UNIT_ASSERT(blocksSubsetLineDataReader.ReadLine(&line));
            UNIT_ASSERT_EQUAL(line, expectedLine);
        }
        UNIT_ASSERT(!blocksSubsetLineDataReader.ReadLine(&line));
    }
}

Y_UNIT_TEST_SUITE(TIndexedSubsetLineDataReaderTest) {
    Y_UNIT_TEST(TestEmptyAll) {
        TTempFile tmpFile(MakeTempName());
        {
            TOFStream out(tmpFile.Name());
        }

        TLineDataReaderArgs lineDataReaderArgs;
        lineDataReaderArgs.PathWithScheme.Path = tmpFile.Name();
        TIndexedSubsetLineDataReader indexedSubsetLineDataReader(
            MakeHolder<TFileLineDataReader>(lineDataReaderArgs),
            TVector<ui64>()
        );

        UNIT_ASSERT(!indexedSubsetLineDataReader.GetHeader().Defined());

        UNIT_ASSERT_EQUAL(indexedSubsetLineDataReader.GetDataLineCount(), 0);

        TString line;
        UNIT_ASSERT(!indexedSubsetLineDataReader.ReadLine(&line));
    }

    Y_UNIT_TEST(TestFullSubset) {
        TTempFile tmpFile(MakeTempName());
        {
            TOFStream out(tmpFile.Name());
            out << "l0\nl1\nl2\nl3\nl4";
        }

        TLineDataReaderArgs lineDataReaderArgs;
        lineDataReaderArgs.PathWithScheme.Path = tmpFile.Name();
        TIndexedSubsetLineDataReader indexedSubsetLineDataReader(
            MakeHolder<TFileLineDataReader>(lineDataReaderArgs),
            TVector<ui64>{0, 1, 2, 3, 4}
        );

        UNIT_ASSERT(!indexedSubsetLineDataReader.GetHeader().Defined());

        UNIT_ASSERT_EQUAL(indexedSubsetLineDataReader.GetDataLineCount(), 5);

        TString line;
        for (auto expectedLine : {"l0", "l1", "l2", "l3", "l4"}) {
            UNIT_ASSERT(indexedSubsetLineDataReader.ReadLine(&line));
            UNIT_ASSERT_EQUAL(line, expectedLine);
        }
        UNIT_ASSERT(!indexedSubsetLineDataReader.ReadLine(&line));
    }

    Y_UNIT_TEST(TestEmptySubset) {
        TTempFile tmpFile(MakeTempName());
        {
            TOFStream out(tmpFile.Name());
            out << "l0\nl1\nl2\nl3\nl4";
        }

        TLineDataReaderArgs lineDataReaderArgs;
        lineDataReaderArgs.PathWithScheme.Path = tmpFile.Name();
        TIndexedSubsetLineDataReader indexedSubsetLineDataReader(
            MakeHolder<TFileLineDataReader>(lineDataReaderArgs),
            TVector<ui64>()
        );

        UNIT_ASSERT(!indexedSubsetLineDataReader.GetHeader().Defined());

        UNIT_ASSERT_EQUAL(indexedSubsetLineDataReader.GetDataLineCount(), 0);

        TString line;
        UNIT_ASSERT(!indexedSubsetLineDataReader.ReadLine(&line));
    }

    Y_UNIT_TEST(TestNontrivialSubset) {
        TTempFile tmpFile(MakeTempName());
        {
            TOFStream out(tmpFile.Name());
            for (auto i : xrange(26)) {
                out << "l" << i << Endl;
            }
        }

        TVector<ui64> subsetIndices = {3, 4, 5, 11, 12, 14, 18, 21, 23};

        TLineDataReaderArgs lineDataReaderArgs;
        lineDataReaderArgs.PathWithScheme.Path = tmpFile.Name();
        TIndexedSubsetLineDataReader indexedSubsetLineDataReader(
            MakeHolder<TFileLineDataReader>(lineDataReaderArgs),
            TVector<ui64>(subsetIndices)
        );

        UNIT_ASSERT(!indexedSubsetLineDataReader.GetHeader().Defined());

        UNIT_ASSERT_EQUAL(indexedSubsetLineDataReader.GetDataLineCount(), subsetIndices.size());

        TString line;
        for (auto subsetIndex : subsetIndices) {
            UNIT_ASSERT(indexedSubsetLineDataReader.ReadLine(&line));
            TString expectedLine = "l" + ToString(subsetIndex);
            UNIT_ASSERT_EQUAL(line, expectedLine);
        }
        UNIT_ASSERT(!indexedSubsetLineDataReader.ReadLine(&line));
    }

    Y_UNIT_TEST(TestSubsetWithRepeatedIndices) {
        TTempFile tmpFile(MakeTempName());
        {
            TOFStream out(tmpFile.Name());
            for (auto i : xrange(26)) {
                out << "l" << i << Endl;
            }
        }

        TVector<ui64> subsetIndices = {3, 4, 5, 11, 11, 12, 12, 12, 14, 18, 21, 23, 23};

        TLineDataReaderArgs lineDataReaderArgs;
        lineDataReaderArgs.PathWithScheme.Path = tmpFile.Name();
        TIndexedSubsetLineDataReader indexedSubsetLineDataReader(
            MakeHolder<TFileLineDataReader>(lineDataReaderArgs),
            TVector<ui64>(subsetIndices)
        );

        UNIT_ASSERT(!indexedSubsetLineDataReader.GetHeader().Defined());

        UNIT_ASSERT_EQUAL(indexedSubsetLineDataReader.GetDataLineCount(), subsetIndices.size());

        TString line;
        for (auto subsetIndex : subsetIndices) {
            UNIT_ASSERT(indexedSubsetLineDataReader.ReadLine(&line));
            TString expectedLine = "l" + ToString(subsetIndex);
            UNIT_ASSERT_EQUAL(line, expectedLine);
        }
        UNIT_ASSERT(!indexedSubsetLineDataReader.ReadLine(&line));
    }
}
