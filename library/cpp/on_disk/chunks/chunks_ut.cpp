#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/file.h>
#include <util/system/filemap.h>
#include <util/system/tempfile.h>

#include "chunked_helpers.h"

/// Data for TChunkedHelpersTest::TestGeneralVector
struct TPodStruct {
    int x;
    float y;
    TPodStruct(int _x = 0, float _y = 0)
        : x(_x)
        , y(_y)
    {
    }
};
/// And its serialization
template <>
struct TSaveLoadVectorNonPodElement<TPodStruct> {
    typedef TPodStruct TItem;
    static inline void Save(IOutputStream* out, const TItem& item) {
        TSerializer<int>::Save(out, item.x);
        TSerializer<float>::Save(out, item.y);
    }

    static inline void Load(IInputStream* in, TItem& item, size_t elementSize) {
        Y_ASSERT(elementSize == sizeof(TItem));
        TSerializer<int>::Load(in, item.x);
        TSerializer<float>::Load(in, item.y);
    }
};

class TChunkedHelpersTest: public TTestBase {
    UNIT_TEST_SUITE(TChunkedHelpersTest);
    UNIT_TEST(TestHash)
    UNIT_TEST(TestGeneralVector)
    UNIT_TEST(TestStrings);
    UNIT_TEST(TestNamedChunkedData);
    UNIT_TEST_SUITE_END();

public:
    void TestHash() {
        {
            TBufferStream stream;
            {
                TPlainHashWriter<ui64, ui16> writer;
                writer.Add(5, 7);
                writer.Save(stream);
            }

            {
                TBlob temp = TBlob::FromStreamSingleThreaded(stream);
                TPlainHash<ui64, ui16> reader(temp);
                ui16 value = 0;
                UNIT_ASSERT(reader.Find(5, &value));
                UNIT_ASSERT_EQUAL(7, value);
                UNIT_ASSERT(!reader.Find(6, &value));
            }
        }

        {
            TBufferStream stream;
            int v = 1;
            wchar16 k = 'a';
            {
                TPlainHashWriter<wchar16, void*> writer;
                writer.Add(k, &v);
                writer.Save(stream);
            }
            {
                TBlob temp = TBlob::FromStreamSingleThreaded(stream);
                TPlainHash<wchar16, void*> reader(temp);
                void* value = nullptr;
                UNIT_ASSERT(reader.Find(k, &value));
                UNIT_ASSERT_EQUAL((int*)value, &v);
            }
        }
    }

    void TestGeneralVector() {
        { /// ui32
            const size_t N = 3;
            TBufferStream stream;
            {
                TGeneralVectorWriter<ui32> writer;
                for (size_t i = 0; i < N; ++i)
                    writer.PushBack(i);
                writer.Save(stream);
            }
            {
                TBlob temp = TBlob::FromStreamSingleThreaded(stream);
                TGeneralVector<ui32> reader(temp);
                UNIT_ASSERT_EQUAL(reader.GetSize(), N);
                for (size_t i = 0; i < N; ++i) {
                    ui32 value;
                    reader.Get(i, value);
                    UNIT_ASSERT_EQUAL(value, i);
                    UNIT_ASSERT_EQUAL(reader.At(i), i);
                }
                UNIT_ASSERT_EQUAL(reader.RealSize(), sizeof(ui64) + N * sizeof(ui32));
            }
        }
        { /// TString
            const size_t N = 4;
            TBufferStream stream;
            {
                TGeneralVectorWriter<TString> writer;
                for (size_t i = 0; i < N; ++i)
                    writer.PushBack(ToString(i));
                writer.Save(stream);
            }
            {
                TBlob temp = TBlob::FromStreamSingleThreaded(stream);
                TGeneralVector<TString> reader(temp);
                UNIT_ASSERT_EQUAL(reader.GetSize(), N);
                for (size_t i = 0; i < N; ++i) {
                    TString value;
                    reader.Get(i, value);
                    UNIT_ASSERT_EQUAL(value, ToString(i));
                    UNIT_ASSERT_EQUAL(reader.Get(i), ToString(i));
                }
                UNIT_ASSERT_EQUAL(reader.RealSize(), sizeof(ui64) * (N + 2) + N * 2);
            }
        }
        { /// some other struct
            typedef TPodStruct TItem;
            const size_t N = 2;
            TBufferStream stream;
            {
                TGeneralVectorWriter<TItem> writer;
                writer.PushBack(TItem(1, 2));
                writer.PushBack(TItem(3, 4));
                writer.Save(stream);
            }
            {
                TBlob temp = TBlob::FromStreamSingleThreaded(stream);
                TGeneralVector<TItem> reader(temp);
                UNIT_ASSERT_EQUAL(reader.GetSize(), N);

                TItem value;
                reader.Get(0, value);
                UNIT_ASSERT(value.x == 1 && value.y == 2.0);

                reader.Get(1, value);
                UNIT_ASSERT(value.x == 3 && value.y == 4.0);

                UNIT_ASSERT_EQUAL(reader.RealSize(), sizeof(ui64) * (N + 2) + N * sizeof(TItem));
            }
        }
        { /// pointer
            const size_t N = 3;
            TVector<int> data_holder(N);
            int* a = &(data_holder[0]);
            TBufferStream stream;
            {
                TGeneralVectorWriter<int*> writer;
                for (size_t i = 0; i < N; ++i) {
                    a[i] = i;
                    writer.PushBack(a + i);
                }
                writer.Save(stream);
            }
            {
                TBlob temp = TBlob::FromStreamSingleThreaded(stream);
                TGeneralVector<int*> reader(temp);
                UNIT_ASSERT_EQUAL(reader.GetSize(), N);
                for (size_t i = 0; i < N; ++i) {
                    int* value;
                    reader.Get(i, value);
                    UNIT_ASSERT_EQUAL(value, a + i);
                    UNIT_ASSERT_EQUAL(reader.At(i), a + i);
                }
                UNIT_ASSERT_EQUAL(reader.RealSize(), sizeof(ui64) + N * sizeof(int*));
            }
        }
        { /// std::pair<int, int>
            typedef std::pair<int, int> TItem;
            const size_t N = 3;
            TBufferStream stream;
            {
                TGeneralVectorWriter<TItem> writer;
                for (size_t i = 0; i < N; ++i)
                    writer.PushBack(TItem(i, i));
                writer.Save(stream);
            }
            {
                TBlob temp = TBlob::FromStreamSingleThreaded(stream);
                TGeneralVector<TItem> reader(temp);
                UNIT_ASSERT_EQUAL(reader.GetSize(), N);
                for (size_t i = 0; i < N; ++i) {
                    TItem value;
                    reader.Get(i, value);
                    UNIT_ASSERT_EQUAL(value, TItem(i, i));
                }
                UNIT_ASSERT_EQUAL(reader.RealSize(), sizeof(ui64) + N * sizeof(TItem));
            }
        }
    }

    void TestStrings() {
        const TString FILENAME = "chunked_helpers_test.bin";
        TTempFileHandle file(FILENAME.c_str());

        {
            TFixedBufferFileOutput fOut(FILENAME);
            TStringsVectorWriter stringsWriter;
            stringsWriter.PushBack("");
            stringsWriter.PushBack("test");
            TChunkedDataWriter writer(fOut);
            WriteBlock(writer, stringsWriter);
            writer.WriteFooter();
        }

        {
            TBlob fIn = TBlob::FromFileSingleThreaded(FILENAME);
            TStringsVector vct(GetBlock(fIn, 0));
            UNIT_ASSERT_EQUAL(vct.Get(0), "");
            UNIT_ASSERT_EQUAL(vct.Get(1), "test");

            bool wasException = false;
            try {
                vct.Get(2);
            } catch (...) {
                wasException = true;
            }
            UNIT_ASSERT(wasException);
        }
    }

    void TestNamedChunkedData() {
        const TString filename = MakeTempName(nullptr, "named_chunked_data_test");
        TTempFile file(filename);

        {
            TFixedBufferFileOutput fOut(filename);
            TNamedChunkedDataWriter writer(fOut);

            writer.NewBlock("alpha");
            writer.Write("123456");

            writer.NewBlock();
            writer.Write("anonymous");

            writer.NewBlock("omega");
            writer.Write("12345678901234567");

            writer.WriteFooter();
        }

        {
            TBlob mf = TBlob::FromFileSingleThreaded(filename);
            TNamedChunkedDataReader reader(mf);

            UNIT_ASSERT(reader.GetBlocksCount() == 3);

            UNIT_ASSERT(reader.HasBlock("alpha"));
            UNIT_ASSERT(reader.HasBlock("omega"));

            UNIT_ASSERT_STRINGS_EQUAL(reader.GetBlockName(0), "alpha");
            UNIT_ASSERT_STRINGS_EQUAL(reader.GetBlockName(1), "");
            UNIT_ASSERT_STRINGS_EQUAL(reader.GetBlockName(2), "omega");

            UNIT_ASSERT_EQUAL(reader.GetBlockLenByName("alpha"), 6); // padding not included
            UNIT_ASSERT_EQUAL(reader.GetBlockLenByName("omega"), 17);

            UNIT_ASSERT(memcmp(reader.GetBlockByName("alpha"), "123456", 6) == 0);
            UNIT_ASSERT(memcmp(reader.GetBlock(1), "anonymous", 9) == 0);
            UNIT_ASSERT(memcmp(reader.GetBlockByName("omega"), "12345678901234567", 17) == 0);
        }
    }
};

UNIT_TEST_SUITE_REGISTRATION(TChunkedHelpersTest);

class TChunkedDataTest: public TTestBase {
private:
    UNIT_TEST_SUITE(TChunkedDataTest);
    UNIT_TEST(Test)
    UNIT_TEST(TestEmpty)
    UNIT_TEST_SUITE_END();

    void Test() {
        TBuffer buffer;
        {
            TBufferOutput out(buffer);
            TChunkedDataWriter writer(out);
            writer.NewBlock();
            writer << "test";
            writer.NewBlock();
            writer << 4;
            writer.NewBlock();
            writer.NewBlock();
            writer << 1;
            writer << 2;
            writer.WriteFooter();
        }
        {
            TBlob blob = TBlob::FromBufferSingleThreaded(buffer);
            TChunkedDataReader data(blob);
            // printf("%d\n", (int)data.GetBlockLen(3));
            UNIT_ASSERT_EQUAL(4, data.GetBlockLen(0));
            UNIT_ASSERT_EQUAL(1, data.GetBlockLen(1));
            UNIT_ASSERT_EQUAL(0, data.GetBlockLen(2));
            UNIT_ASSERT_EQUAL(2, data.GetBlockLen(3));
        }
    }

    void TestEmpty() {
        TBuffer buffer;
        {
            TBufferOutput out(buffer);
            TChunkedDataWriter writer(out);
            writer.NewBlock();
            writer.NewBlock();
            writer.WriteFooter();
        }
        {
            TBlob blob = TBlob::FromBufferSingleThreaded(buffer);
            TChunkedDataReader data(blob);
            // printf("%d\n", (int)data.GetBlockLen(1));
            UNIT_ASSERT_EQUAL(0, data.GetBlockLen(0));
            UNIT_ASSERT_EQUAL(0, data.GetBlockLen(1));
        }
    }
};

UNIT_TEST_SUITE_REGISTRATION(TChunkedDataTest);
