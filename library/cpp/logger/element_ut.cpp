#include "log.h"
#include "element.h"
#include "stream.h"

#include <util/generic/string.h>
#include <util/stream/str.h>
#include <util/generic/ptr.h>
#include <utility>

#include <library/cpp/testing/unittest/registar.h>


class TLogElementTest: public TTestBase {
    UNIT_TEST_SUITE(TLogElementTest);
    UNIT_TEST(TestMoveCtor);
    UNIT_TEST(TestWith);
    UNIT_TEST_SUITE_END();

public:
    void TestMoveCtor();
    void TestWith();
};

UNIT_TEST_SUITE_REGISTRATION(TLogElementTest);

void TLogElementTest::TestMoveCtor() {
    TStringStream output;
    TLog log(MakeHolder<TStreamLogBackend>(&output));

    THolder<TLogElement> src = MakeHolder<TLogElement>(&log);

    TString message = "Hello, World!";
    (*src) << message;

    THolder<TLogElement> dst = MakeHolder<TLogElement>(std::move(*src));

    src.Destroy();
    UNIT_ASSERT(output.Str() == "");

    dst.Destroy();
    UNIT_ASSERT(output.Str() == message);
}

void TLogElementTest::TestWith() {
    TStringStream output;
    TLog log(MakeHolder<TStreamWithContextLogBackend>(&output));

    THolder<TLogElement> src = MakeHolder<TLogElement>(&log);

    TString message = "Hello, World!";
    (*src).With("Foo", "Bar").With("Foo", "Baz") << message;

    src.Destroy();
    UNIT_ASSERT(output.Str() == "Hello, World!; Foo=Bar; Foo=Baz; ");
}
