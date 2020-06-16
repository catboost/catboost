#include <library/cpp/resource/resource.h>
#include <library/cpp/testing/unittest/registar.h>

#include <tools/enum_parser/parse_enum/parse_enum.h>

typedef TEnumParser::TEnum TEnum;
typedef TEnumParser::TEnums TEnums;
typedef TEnumParser::TItems TItems;

Y_UNIT_TEST_SUITE(TEnumParserTest) {

    Y_UNIT_TEST(MainTest) {
        TString text = NResource::Find("/enums");
        TMemoryInput input(text.data(), text.size());
        TEnumParser parser(input);
        const TEnums& enums = parser.Enums;

        UNIT_ASSERT_VALUES_EQUAL(enums.size(), 16u);

        // check ESimple
        {
            const TEnum& e = enums[0];
            UNIT_ASSERT_VALUES_EQUAL(e.Scope.size(), 0u);
            UNIT_ASSERT_VALUES_EQUAL(e.CppName, "ESimple");
            const TItems& it = e.Items;
            UNIT_ASSERT_VALUES_EQUAL(it.size(), 3u);

            UNIT_ASSERT_VALUES_EQUAL(it[0].CppName, "Http");
            UNIT_ASSERT_VALUES_EQUAL(it[0].Aliases.size(), 0u);
            UNIT_ASSERT(!it[0].Value.Defined());

            UNIT_ASSERT_VALUES_EQUAL(it[1].CppName, "Https");
            UNIT_ASSERT_VALUES_EQUAL(it[1].Aliases.size(), 0u);
            UNIT_ASSERT(!it[1].Value.Defined());

            UNIT_ASSERT_VALUES_EQUAL(it[2].CppName, "ItemCount");
            UNIT_ASSERT_VALUES_EQUAL(it[2].Aliases.size(), 0u);
            UNIT_ASSERT(!it[2].Value.Defined());
        }

        // ESimpleWithComma
        {
            const TEnum& e = enums[1];
            UNIT_ASSERT_VALUES_EQUAL(e.Scope.size(), 0u);
            UNIT_ASSERT_VALUES_EQUAL(e.CppName, "ESimpleWithComma");
            const TItems& it = e.Items;
            UNIT_ASSERT_VALUES_EQUAL(it.size(), 4u);

            UNIT_ASSERT_VALUES_EQUAL(it[0].CppName, "Http");
            UNIT_ASSERT(it[0].Value.Defined());
            UNIT_ASSERT_VALUES_EQUAL(*it[0].Value, "3");
            UNIT_ASSERT_VALUES_EQUAL(it[0].Aliases.size(), 0u);

            UNIT_ASSERT_VALUES_EQUAL(it[1].CppName, "Http2");
            UNIT_ASSERT(it[1].Value.Defined());
            UNIT_ASSERT_VALUES_EQUAL(it[1].Aliases.size(), 0u);
            UNIT_ASSERT_VALUES_EQUAL(*it[1].Value, "Http");

            UNIT_ASSERT_VALUES_EQUAL(it[2].CppName, "Https");
            UNIT_ASSERT_VALUES_EQUAL(it[2].Aliases.size(), 0u);
            UNIT_ASSERT(!it[2].Value.Defined());

            UNIT_ASSERT_VALUES_EQUAL(it[3].CppName, "ItemCount");
            UNIT_ASSERT_VALUES_EQUAL(it[3].Aliases.size(), 0u);
            UNIT_ASSERT(!it[3].Value.Defined());
        }

        // check ECustomAliases
        {
            const TEnum& e = enums[2];
            UNIT_ASSERT_VALUES_EQUAL(e.Scope.size(), 0u);
            UNIT_ASSERT_VALUES_EQUAL(e.CppName, "ECustomAliases");
            const TItems& it = e.Items;
            UNIT_ASSERT_VALUES_EQUAL(it.size(), 3u);
            UNIT_ASSERT_VALUES_EQUAL(it[0].CppName, "CAHttp");
            UNIT_ASSERT(it[0].Value.Defined());
            UNIT_ASSERT_VALUES_EQUAL(*it[0].Value, "3");
            UNIT_ASSERT_VALUES_EQUAL(it[0].Aliases.size(), 1u);
            UNIT_ASSERT_VALUES_EQUAL(it[0].Aliases[0], "http");

            UNIT_ASSERT(!it[1].Value.Defined());
            UNIT_ASSERT_VALUES_EQUAL(it[1].CppName, "CAHttps");
            UNIT_ASSERT_VALUES_EQUAL(it[1].Aliases.size(), 1u);
            UNIT_ASSERT_VALUES_EQUAL(it[1].Aliases[0], "https");

            UNIT_ASSERT_VALUES_EQUAL(it[2].CppName, "CAItemCount");
            UNIT_ASSERT_VALUES_EQUAL(it[2].Aliases.size(), 0u);
        }

        // check EMultipleAliases
        {
            const TEnum& e = enums[3];
            UNIT_ASSERT_VALUES_EQUAL(e.Scope.size(), 0u);
            UNIT_ASSERT_VALUES_EQUAL(e.CppName, "EMultipleAliases");
            const TItems& it = e.Items;
            UNIT_ASSERT_VALUES_EQUAL(it.size(), 3u);
            UNIT_ASSERT_VALUES_EQUAL(it[0].CppName, "MAHttp");
            UNIT_ASSERT(it[0].Value.Defined());
            UNIT_ASSERT_VALUES_EQUAL(*it[0].Value, "9");
            UNIT_ASSERT_VALUES_EQUAL(it[0].Aliases.size(), 3u);
            UNIT_ASSERT_VALUES_EQUAL(it[0].Aliases[0], "http://");
            UNIT_ASSERT_VALUES_EQUAL(it[0].Aliases[1], "secondary");
            // yes, quoted values are NOT decoded, it is a known (minor) bug
            UNIT_ASSERT_VALUES_EQUAL(it[0].Aliases[2], "old\\nvalue");

            UNIT_ASSERT_VALUES_EQUAL(it[1].CppName, "MAHttps");
            UNIT_ASSERT(it[1].Value.Defined());
            UNIT_ASSERT_VALUES_EQUAL(*it[1].Value, "1");
            UNIT_ASSERT_VALUES_EQUAL(it[1].Aliases.size(), 1u);
            UNIT_ASSERT_VALUES_EQUAL(it[1].Aliases[0], "https://");

            UNIT_ASSERT_VALUES_EQUAL(it[2].CppName, "MAItemCount");
            UNIT_ASSERT(!it[2].Value.Defined());
            UNIT_ASSERT_VALUES_EQUAL(it[2].Aliases.size(), 0u);
        }

        // check NEnumNamespace::EInNamespace
        {
            const TEnum& e = enums[4];
            UNIT_ASSERT_VALUES_EQUAL(e.Scope.size(), 1u);
            UNIT_ASSERT_VALUES_EQUAL(e.Scope[0], "NEnumNamespace");
            UNIT_ASSERT_VALUES_EQUAL(e.CppName, "EInNamespace");
            const TItems& it = e.Items;
            UNIT_ASSERT_VALUES_EQUAL(it.size(), 3u);
            UNIT_ASSERT_VALUES_EQUAL(it[0].CppName, "Http");
            UNIT_ASSERT(it[0].Value.Defined());
        }

        // check NEnumNamespace::TEnumClass::EInClass
        {
            const TEnum& e = enums[5];
            UNIT_ASSERT_VALUES_EQUAL(e.Scope.size(), 2u);
            UNIT_ASSERT_VALUES_EQUAL(e.Scope[0], "NEnumNamespace");
            UNIT_ASSERT_VALUES_EQUAL(e.Scope[1], "TEnumClass");
            UNIT_ASSERT_VALUES_EQUAL(e.CppName, "EInClass");
            const TItems& it = e.Items;
            UNIT_ASSERT_VALUES_EQUAL(it.size(), 3u);
            UNIT_ASSERT_VALUES_EQUAL(it[0].CppName, "Http");
            UNIT_ASSERT(it[0].Value.Defined());
            UNIT_ASSERT_VALUES_EQUAL(*it[0].Value, "9");

            UNIT_ASSERT(it[1].Value.Defined());
            UNIT_ASSERT_VALUES_EQUAL(*it[1].Value, "NEnumNamespace::Https");

            UNIT_ASSERT_VALUES_EQUAL(it[2].CppName, "Https3");
            UNIT_ASSERT(it[2].Value.Defined());
            UNIT_ASSERT_VALUES_EQUAL(*it[2].Value, "1  + 2");
        }

        // check unnamed enum (no code should be generated for it)
        {
            const TEnum& e = enums[6];
            UNIT_ASSERT_VALUES_EQUAL(e.Scope.size(), 0u);
            UNIT_ASSERT_VALUES_EQUAL(e.CppName, "");
            const TItems& it = e.Items;
            UNIT_ASSERT_VALUES_EQUAL(it.size(), 3u);
        }

        // TEXT_WEIGHT
        {
            const TEnum& e = enums[7];
            UNIT_ASSERT_VALUES_EQUAL(e.Scope.size(), 0u);
            UNIT_ASSERT_VALUES_EQUAL(e.CppName, "TEXT_WEIGHT");
            const TItems& it = e.Items;
            UNIT_ASSERT_VALUES_EQUAL(it.size(), 5u);

            UNIT_ASSERT_VALUES_EQUAL(it[0].CppName, "WEIGHT_ZERO");
            UNIT_ASSERT(it[0].Value.Defined());
            UNIT_ASSERT_VALUES_EQUAL(*it[0].Value, "-1");
            UNIT_ASSERT_VALUES_EQUAL(it[0].Aliases.size(), 0u);

            UNIT_ASSERT_VALUES_EQUAL(it[1].CppName, "WEIGHT_LOW");
            UNIT_ASSERT_VALUES_EQUAL(it[1].Aliases.size(), 0u);
            UNIT_ASSERT(!it[1].Value.Defined());

            UNIT_ASSERT_VALUES_EQUAL(it[2].CppName, "WEIGHT_NORMAL");
            UNIT_ASSERT_VALUES_EQUAL(it[2].Aliases.size(), 0u);
            UNIT_ASSERT(!it[2].Value.Defined());
        }

        // EDuplicateKeys
        {
            const TEnum& e = enums[8];
            UNIT_ASSERT_VALUES_EQUAL(e.Scope.size(), 0u);
            UNIT_ASSERT_VALUES_EQUAL(e.CppName, "EDuplicateKeys");
            const TItems& it = e.Items;
            UNIT_ASSERT_VALUES_EQUAL(it.size(), 5u);

            UNIT_ASSERT_VALUES_EQUAL(it[0].CppName, "Key0");
            UNIT_ASSERT(it[0].Value.Defined());
            UNIT_ASSERT_VALUES_EQUAL(*it[0].Value, "0");
            UNIT_ASSERT_VALUES_EQUAL(it[0].Aliases.size(), 0u);

            UNIT_ASSERT_VALUES_EQUAL(it[1].CppName, "Key0Second");
            UNIT_ASSERT(it[1].Value.Defined());
            UNIT_ASSERT_VALUES_EQUAL(*it[1].Value, "Key0");
            UNIT_ASSERT_VALUES_EQUAL(it[1].Aliases.size(), 0u);
        }

        // EEmpty
        {
            const TEnum& e = enums[10];
            const TItems& it = e.Items;
            UNIT_ASSERT_VALUES_EQUAL(it.size(), 0u);
        }

        // NComposite::NInner::EInCompositeNamespaceSimple
        {
            const TEnum& e = enums[11];
            UNIT_ASSERT_VALUES_EQUAL(e.Scope.size(), 1u);
            UNIT_ASSERT_VALUES_EQUAL(e.Scope[0], "NComposite::NInner");
            UNIT_ASSERT_VALUES_EQUAL(e.CppName, "EInCompositeNamespaceSimple");
            const TItems& it = e.Items;
            UNIT_ASSERT_VALUES_EQUAL(it.size(), 3u);
            UNIT_ASSERT_VALUES_EQUAL(it[0].CppName, "one");
            UNIT_ASSERT_VALUES_EQUAL(*it[1].Value, "2") ;
        }

        // NOuterSimple::NComposite::NMiddle::NInner::NInnerSimple::TEnumClass::EVeryDeep
        {
            const TEnum& e = enums[12];
            UNIT_ASSERT_VALUES_EQUAL(e.Scope.size(), 4u);
            UNIT_ASSERT_VALUES_EQUAL(e.Scope[0], "NOuterSimple");
            UNIT_ASSERT_VALUES_EQUAL(e.Scope[1], "NComposite::NMiddle::NInner");
            UNIT_ASSERT_VALUES_EQUAL(e.Scope[2], "NInnerSimple");
            UNIT_ASSERT_VALUES_EQUAL(e.Scope[3], "TEnumClass");
            UNIT_ASSERT_VALUES_EQUAL(e.CppName, "EVeryDeep");
            const TItems& it = e.Items;
            UNIT_ASSERT_VALUES_EQUAL(it.size(), 2u);
            UNIT_ASSERT_VALUES_EQUAL(it[0].CppName, "Key0");
            UNIT_ASSERT_VALUES_EQUAL(it[1].CppName, "Key1");
            UNIT_ASSERT_VALUES_EQUAL(*it[1].Value, "1");
        }

        // ENonLiteralValues
        {
            const TEnum& e = enums[13];
            UNIT_ASSERT_VALUES_EQUAL(e.Scope.size(), 0u);
            UNIT_ASSERT_VALUES_EQUAL(e.CppName, "ENonLiteralValues");
            const TItems& it = e.Items;
            UNIT_ASSERT_VALUES_EQUAL(it.size(), 5u);
            UNIT_ASSERT_VALUES_EQUAL(it[0].CppName, "one");
            UNIT_ASSERT_VALUES_EQUAL(*it[0].Value, "MACRO(1, 2)");
            UNIT_ASSERT_VALUES_EQUAL(it[1].CppName, "two");
            UNIT_ASSERT_VALUES_EQUAL(*it[1].Value, "2");
            UNIT_ASSERT_VALUES_EQUAL(it[2].CppName, "three");
            UNIT_ASSERT_VALUES_EQUAL(*it[2].Value, "func(3)");
            UNIT_ASSERT_VALUES_EQUAL(it[3].CppName, "four");
            UNIT_ASSERT_VALUES_EQUAL(it[3].Value.Defined(), false);
            UNIT_ASSERT_VALUES_EQUAL(it[4].CppName, "five");
            UNIT_ASSERT_VALUES_EQUAL(it[4].Value, "MACRO(MACRO(1, 2), 2)");
        }

        // NotifyingStatus
        {
            const TEnum& e = enums[15];
            UNIT_ASSERT_VALUES_EQUAL(e.Scope.size(), 0u);
            UNIT_ASSERT_VALUES_EQUAL(e.CppName, "NotifyingStatus");
            const TItems& it = e.Items;
            UNIT_ASSERT_VALUES_EQUAL(it.size(), 4u);
            UNIT_ASSERT_VALUES_EQUAL(it[0].CppName, "NEW");
            UNIT_ASSERT_VALUES_EQUAL(*it[0].Value, "0");
            UNIT_ASSERT_VALUES_EQUAL(it[1].CppName, "FAILED_WILL_RETRY");
            UNIT_ASSERT_VALUES_EQUAL(*it[1].Value, "1");
            UNIT_ASSERT_VALUES_EQUAL(it[2].CppName, "FAILED_NO_MORE_TRIALS");
            UNIT_ASSERT_VALUES_EQUAL(*it[2].Value, "2");
            UNIT_ASSERT_VALUES_EQUAL(it[3].CppName, "SENT");
            UNIT_ASSERT_VALUES_EQUAL(*it[3].Value, "3");
        }
    }

    Y_UNIT_TEST(BadCodeParseTest) {
        TString text = NResource::Find("/badcode");
        TMemoryInput input(text.data(), text.size());
        TEnumParser parser(input);
        const TEnums& enums = parser.Enums;

        UNIT_ASSERT_VALUES_EQUAL(enums.size(), 1u);

        // check <anonymous namespace>::ETest correct parsing
        {
            const TEnum& e = enums[0];
            UNIT_ASSERT_VALUES_EQUAL(e.Scope.size(), 1u);
            UNIT_ASSERT_VALUES_EQUAL(e.CppName, "ETest");
            const TItems& it = e.Items;
            UNIT_ASSERT_VALUES_EQUAL(it.size(), 3u);
            UNIT_ASSERT_VALUES_EQUAL(it[0].CppName, "Http");
            UNIT_ASSERT(it[0].Value.Defined());
        }

    }

    Y_UNIT_TEST(UnbalancedCodeParseTest) {
        // Thanks gotmanov@ for providing this example
        TString text = NResource::Find("/unbalanced");
        TMemoryInput input(text.data(), text.size());
        try {
            TEnumParser parser(input);
            UNIT_ASSERT(false);
        } catch(...) {
            UNIT_ASSERT(CurrentExceptionMessage().Contains("unbalanced scope. Did you miss a closing"));
        }
    }

    Y_UNIT_TEST(AliasBeforeNameTest) {
        TString text = NResource::Find("/alias_before_name");
        TMemoryInput input(text.data(), text.size());
        try {
            TEnumParser parser(input);
            UNIT_ASSERT(false);
        } catch(...) {
            UNIT_ASSERT(CurrentExceptionMessage().Contains("https://clubs.at.yandex-team.ru/stackoverflow/2603"));
        }
    }
}
