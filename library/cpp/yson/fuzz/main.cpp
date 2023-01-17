#include <library/cpp/yson/parser.h>
#include <library/cpp/yson/consumer.h>

#include <util/generic/string.h>
#include <util/generic/strbuf.h>

class TDummyConsumer: public NYson::TYsonConsumerBase {
public:
    void OnStringScalar(TStringBuf) override {}
    void OnInt64Scalar(i64) override {}
    void OnUint64Scalar(ui64) override {}
    void OnDoubleScalar(double) override {}
    void OnBooleanScalar(bool) override {}
    void OnEntity() override {}
    void OnBeginList() override {}
    void OnListItem() override {}
    void OnEndList() override {}
    void OnBeginMap() override {}
    void OnKeyedItem(TStringBuf) override {}
    void OnEndMap() override {}
    void OnBeginAttributes() override {}
    void OnEndAttributes() override {}
};

extern "C" int LLVMFuzzerTestOneInput(char *data, size_t size) {
    TStringBuf yson{data, size};
    TDummyConsumer consumer;

    try {
        ParseYsonStringBuffer(yson, &consumer);
    } catch (const yexception& e) {
        Cout << "Exception: " << e.what() << Endl;
    }
    
    return 0;
}

