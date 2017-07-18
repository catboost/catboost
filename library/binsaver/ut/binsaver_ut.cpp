#include <library/binsaver/bin_saver.h>
#include <library/binsaver/util_stream_io.h>
#include <library/unittest/registar.h>

#include <util/stream/buffer.h>
#include <util/generic/map.h>

struct TBinarySerializable {
    ui32 Data = 0;
};

struct TNonBinarySerializable {
    ui32 Data = 0;
    TString StrData;
};

struct TCustomSerializer {
    ui32 Data = 0;
    TString StrData;
    SAVELOAD(StrData, Data);
};

struct TCustomOuterSerializer {
    ui32 Data = 0;
    TString StrData;
};

void operator & (TCustomOuterSerializer& s, IBinSaver& f);

struct TCustomOuterSerializerTmpl {
    ui32 Data = 0;
    TString StrData;
};

struct TCustomOuterSerializerTmplDerived : public TCustomOuterSerializerTmpl {
    ui32 Data = 0;
    TString StrData;
};

struct TMoveOnlyType {
    ui32 Data = 0;

    TMoveOnlyType() = default;
    TMoveOnlyType(TMoveOnlyType&&) = default;

    bool operator ==(const TMoveOnlyType& obj) const {
        return Data == obj.Data;
    }
};

struct TTypeWithArray {
    ui32 Data = 1;
    TString Array[2][2]{{"test", "data"}, {"and", "more"}};

    SAVELOAD(Data, Array);
    bool operator ==(const TTypeWithArray& obj) const {
        return Data == obj.Data
            && std::equal(std::begin(Array[0]), std::end(Array[0]), obj.Array[0])
            && std::equal(std::begin(Array[1]), std::end(Array[1]), obj.Array[1]);
    }
};

template<typename T, typename = std::enable_if_t<std::is_base_of<TCustomOuterSerializerTmpl, T>::value>>
int operator & (T& s, IBinSaver& f);

static bool operator==(const TBlob& l, const TBlob& r) {
    return TStringBuf(~l, +l) == TStringBuf(~r, +r);
}

SIMPLE_UNIT_TEST_SUITE(BinSaver) {
    SIMPLE_UNIT_TEST(HasTrivialSerializer) {
        UNIT_ASSERT( !IBinSaver::HasNonTrivialSerializer<TBinarySerializable>(0u) );
        UNIT_ASSERT( !IBinSaver::HasNonTrivialSerializer<TNonBinarySerializable>(0u) );
        UNIT_ASSERT( IBinSaver::HasNonTrivialSerializer<TCustomSerializer>(0u) );
        UNIT_ASSERT( IBinSaver::HasNonTrivialSerializer<TCustomOuterSerializer>(0u) );
        UNIT_ASSERT( IBinSaver::HasNonTrivialSerializer<TCustomOuterSerializerTmpl>(0u) );
        UNIT_ASSERT( IBinSaver::HasNonTrivialSerializer<TCustomOuterSerializerTmplDerived>(0u) );
        UNIT_ASSERT( IBinSaver::HasNonTrivialSerializer<yvector<TCustomSerializer>>(0u) );
    }

    template<typename T>
    void TestSerialization(const T& original) {
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
        UNIT_ASSERT_EQUAL(original, restored);
    }

    SIMPLE_UNIT_TEST(TestStroka) {
        TestSerialization(TString("QWERTY"));
    }

    SIMPLE_UNIT_TEST(TestMoveOnlyType) {
        TestSerialization(TMoveOnlyType());
    }

    SIMPLE_UNIT_TEST(TestVectorStrok) {
        TestSerialization(yvector<TString>{"A", "B", "C"});
    }

    SIMPLE_UNIT_TEST(TestCArray) {
        TestSerialization(TTypeWithArray());
    }

    SIMPLE_UNIT_TEST(TestSets) {
        TestSerialization(yhash_set<TString>{"A", "B", "C"});
        TestSerialization(yset<TString>{"A", "B", "C"});
    }

    SIMPLE_UNIT_TEST(TestMaps) {
        TestSerialization(yhash<TString, ui32>{{"A", 1}, {"B", 2}, {"C", 3}});
        TestSerialization(ymap<TString, ui32>{{"A", 1}, {"B", 2}, {"C", 3}});
    }

    SIMPLE_UNIT_TEST(TestBlob) {
        TestSerialization(TBlob::FromStringSingleThreaded("qwerty"));
    }

    SIMPLE_UNIT_TEST(TestPod) {
        struct TPod {
            ui32 A = 5;
            ui64 B = 7;
            bool operator == (const TPod& other) const {
                return A == other.A && B == other.B;
            }
        };
        TestSerialization(TPod());
        TPod custom;
        custom.A = 25;
        custom.B = 37;
        TestSerialization(custom);
        TestSerialization(yvector<TPod>{ custom });
    }

    SIMPLE_UNIT_TEST(TestSubPod) {
        struct TPod {
            struct TSub {
                ui32 X = 10;
                bool operator == (const TSub& other) const {
                    return X == other.X;
                }
            };
            yvector<TSub> B;
            int operator & (IBinSaver& f) {
                f.Add(0, &B);
                return 0;
            }
            bool operator == (const TPod& other) const {
                return B == other.B;
            }
        };
        TestSerialization(TPod());
        TPod::TSub sub;
        sub.X = 1;
        TPod custom;
        custom.B = { sub };
        TestSerialization(yvector<TPod>{ custom });
    }


    SIMPLE_UNIT_TEST(TestMemberAndOpIsMain) {
        struct TBase {
            TString S;
            virtual int operator & (IBinSaver& f) {
                f.Add(0, &S);
                return 0;
            }
            virtual ~TBase() = default;
        };

        struct TDerived : public TBase {
            int A = 0;
            int operator & (IBinSaver& f) override {
                f.Add(0, static_cast<TBase*>(this));
                f.Add(0, &A);
                return 0;
            }
            bool operator == (const TDerived& other) const {
                return A == other.A && S == other.S;
            }
        };

        TDerived obj;
        obj.S = "TString";
        obj.A = 42;

        TestSerialization(obj);
    }
};

