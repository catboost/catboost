#include <catboost/libs/helpers/xml_output.h>

#include <util/stream/str.h>

#include <library/cpp/testing/unittest/registar.h>


Y_UNIT_TEST_SUITE(XmlOutput) {
    Y_UNIT_TEST(WriteXmlEscaped) {
        {
            TStringStream out;
            WriteXmlEscaped("", &out);
            UNIT_ASSERT_VALUES_EQUAL(out.Str(), "");
        }
        {
            TStringStream out;
            WriteXmlEscaped("simple text", &out);
            UNIT_ASSERT_VALUES_EQUAL(out.Str(), "simple text");
        }
        {
            TStringStream out;
            WriteXmlEscaped("text with &special' chars", &out);
            UNIT_ASSERT_VALUES_EQUAL(out.Str(), "text with &amp;special&apos; chars");
        }
        {
            TStringStream out;
            WriteXmlEscaped("<?xml version=\"1.0\"?>", &out);
            UNIT_ASSERT_VALUES_EQUAL(out.Str(), "&lt;?xml version=&quot;1.0&quot;?&gt;");
        }
    }

    Y_UNIT_TEST(XmlOutputContext) {
        {
            TStringStream out;
            {
                TXmlOutputContext xmlOutput(&out, "root");
            }
            UNIT_ASSERT_VALUES_EQUAL(out.Str(), "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<root/>\n");
        }
        {
            TStringStream out;
            {
                TXmlOutputContext xmlOutput(&out, "root");
                xmlOutput.AddAttr("a", "b");
                xmlOutput.AddAttr("c", 12);
            }
            UNIT_ASSERT_VALUES_EQUAL(out.Str(), "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<root a=\"b\" c=\"12\"/>\n");
        }
        {
            TStringStream out;
            {
                TXmlOutputContext xmlOutput(&out, "root");
                xmlOutput.AddAttr("a", "b");
                xmlOutput.AddAttr("c", 12);
                {
                    TXmlElementOutputContext tag1(&xmlOutput, "tag1");
                }
                {
                    TXmlElementOutputContext tag2(&xmlOutput, "tag2");
                    xmlOutput.AddAttr("at2", "val2");
                }
                {
                    TXmlElementOutputContext tag3(&xmlOutput, "tag3");
                    xmlOutput.AddAttr("at3", 31);
                    {
                        TXmlElementOutputContext tag31(&xmlOutput, "tag31");
                        xmlOutput.GetOutput() << "<text31>";
                        {
                            TXmlElementOutputContext tag331(&xmlOutput, "tag331");
                            xmlOutput.AddAttr("atx1", 0.1);
                            xmlOutput.AddAttr("atx2", (const char*)"text_\"x2\"");
                        }
                    }
                }
            }
            UNIT_ASSERT_VALUES_EQUAL(
                out.Str(),
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                "<root a=\"b\" c=\"12\">\n"
                "<tag1/>\n"
                "<tag2 at2=\"val2\"/>\n"
                "<tag3 at3=\"31\">\n"
                "<tag31>&lt;text31&gt;<tag331 atx1=\"0.1\" atx2=\"text_&quot;x2&quot;\"/>\n"
                "</tag31>\n"
                "</tag3>\n"
                "</root>\n");
        }
    }
}
