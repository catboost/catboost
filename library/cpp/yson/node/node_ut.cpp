#include "node.h"
#include "node_io.h"

#include <library/cpp/unittest/registar.h>

#include <util/ysaveload.h>

using namespace NYT;

template<>
void Out<NYT::TNode>(IOutputStream& s, const NYT::TNode& node)
{
    s << "TNode:" << NodeToYsonString(node);
}

Y_UNIT_TEST_SUITE(YtNodeTest) {
    Y_UNIT_TEST(TestConstsructors) {
        TNode nodeEmpty;
        UNIT_ASSERT_EQUAL(nodeEmpty.GetType(), TNode::Undefined);

        TNode nodeString("foobar");
        UNIT_ASSERT_EQUAL(nodeString.GetType(), TNode::String);
        UNIT_ASSERT(nodeString.IsString());
        UNIT_ASSERT_VALUES_EQUAL(nodeString.AsString(), "foobar");

        TNode nodeInt(int(54));
        UNIT_ASSERT_EQUAL(nodeInt.GetType(), TNode::Int64);
        UNIT_ASSERT(nodeInt.IsInt64());
        UNIT_ASSERT(!nodeInt.IsUint64());
        UNIT_ASSERT_VALUES_EQUAL(nodeInt.AsInt64(), 54ull);

        TNode nodeUint(ui64(42));
        UNIT_ASSERT_EQUAL(nodeUint.GetType(), TNode::Uint64);
        UNIT_ASSERT(nodeUint.IsUint64());
        UNIT_ASSERT(!nodeUint.IsInt64());
        UNIT_ASSERT_VALUES_EQUAL(nodeUint.AsUint64(), 42ull);

        TNode nodeDouble(double(2.3));
        UNIT_ASSERT_EQUAL(nodeDouble.GetType(), TNode::Double);
        UNIT_ASSERT(nodeDouble.IsDouble());
        UNIT_ASSERT_VALUES_EQUAL(nodeDouble.AsDouble(), double(2.3));

        TNode nodeBool(true);
        UNIT_ASSERT_EQUAL(nodeBool.GetType(), TNode::Bool);
        UNIT_ASSERT(nodeBool.IsBool());
        UNIT_ASSERT_VALUES_EQUAL(nodeBool.AsBool(), true);

        TNode nodeEntity = TNode::CreateEntity();
        UNIT_ASSERT_EQUAL(nodeEntity.GetType(), TNode::Null);
        UNIT_ASSERT(nodeEntity.IsEntity());
    }

    Y_UNIT_TEST(TestPredicates) {
        const TNode undefinedNode;
        UNIT_ASSERT(undefinedNode.IsUndefined());
        UNIT_ASSERT(!undefinedNode.IsNull());
        UNIT_ASSERT(!undefinedNode.HasValue());

        const TNode nullNode = TNode::CreateEntity();
        UNIT_ASSERT(!nullNode.IsUndefined());
        UNIT_ASSERT(nullNode.IsNull());
        UNIT_ASSERT(!nullNode.HasValue());

        const TNode intNode(int(64));
        UNIT_ASSERT(!intNode.IsUndefined());
        UNIT_ASSERT(!intNode.IsNull());
        UNIT_ASSERT(intNode.HasValue());

        const TNode stringNode("blah");
        UNIT_ASSERT(!stringNode.IsUndefined());
        UNIT_ASSERT(!stringNode.IsNull());
        UNIT_ASSERT(stringNode.HasValue());
    }

    Y_UNIT_TEST(TestComplexConstructors) {
        const TNode listNode = TNode::CreateList({"one", 2, "tree"});
        const auto expectedListValue = std::vector<TNode>({"one", 2, "tree"});
        UNIT_ASSERT_VALUES_EQUAL(listNode.AsList(), expectedListValue);

        const TNode mapNode = TNode::CreateMap({{"one", 1}, {"two", 2u}});
        const auto expectedMapValue = THashMap<TString, TNode>({{"one", 1}, {"two", 2u}});
        UNIT_ASSERT_VALUES_EQUAL(mapNode.AsMap(), expectedMapValue);
    }

    Y_UNIT_TEST(TestNodeMap) {
        TNode nodeMap = TNode()("foo", "bar")("bar", "baz");
        UNIT_ASSERT(nodeMap.IsMap());
        UNIT_ASSERT_EQUAL(nodeMap.GetType(), TNode::Map);
        UNIT_ASSERT_VALUES_EQUAL(nodeMap.Size(), 2);

        UNIT_ASSERT(nodeMap.HasKey("foo"));
        UNIT_ASSERT(!nodeMap.HasKey("42"));
        UNIT_ASSERT_EQUAL(nodeMap["foo"], TNode("bar"));
        UNIT_ASSERT_EQUAL(nodeMap["bar"], TNode("baz"));

        // const version of operator[]
        UNIT_ASSERT_EQUAL(static_cast<const TNode&>(nodeMap)["42"].GetType(), TNode::Undefined);
        UNIT_ASSERT(!nodeMap.HasKey("42"));

        // nonconst version of operator[]
        UNIT_ASSERT_EQUAL(nodeMap["42"].GetType(), TNode::Undefined);
        UNIT_ASSERT(nodeMap.HasKey("42"));

        nodeMap("rock!!!", TNode()
            ("Pink", "Floyd")
            ("Purple", "Deep"));

        TNode copyNode;
        copyNode = nodeMap;
        UNIT_ASSERT_EQUAL(copyNode["foo"], TNode("bar"));
        UNIT_ASSERT_EQUAL(copyNode["bar"], TNode("baz"));
        UNIT_ASSERT(copyNode["42"].GetType() == TNode::Undefined);
        UNIT_ASSERT_EQUAL(copyNode["rock!!!"]["Purple"], TNode("Deep"));
    }

    Y_UNIT_TEST(TestNodeList) {
        TNode nodeList = TNode().Add("foo").Add(42).Add(3.14);
        UNIT_ASSERT(nodeList.IsList());
        UNIT_ASSERT_EQUAL(nodeList.GetType(), TNode::List);
        UNIT_ASSERT_VALUES_EQUAL(nodeList.Size(), 3);

        UNIT_ASSERT_EQUAL(nodeList[1], TNode(42));
        nodeList.Add(TNode().Add("ls").Add("pwd"));

        TNode copyNode;
        copyNode = nodeList;
        UNIT_ASSERT_EQUAL(copyNode[0], TNode("foo"));
        UNIT_ASSERT_EQUAL(copyNode[3][1], TNode("pwd"));
    }

    Y_UNIT_TEST(TestInsertingMethodsFromTemporaryObjects) {
        // check that .Add(...) doesn't return lvalue reference to temporary object
        {
            const TNode& nodeList = TNode().Add(0).Add("pass").Add(0);
            UNIT_ASSERT_EQUAL(nodeList[1], TNode("pass"));
        }

        // check that .operator()(...) doesn't return lvalue reference to temporary object
        {
            const TNode& nodeMap = TNode()("1", 0)("2", "pass")("3", 0);
            UNIT_ASSERT_EQUAL(nodeMap["2"], TNode("pass"));
        }
    }

    Y_UNIT_TEST(TestAttributes) {
        TNode node = TNode()("lee", 42)("faa", 54);
        UNIT_ASSERT(!node.HasAttributes());
        node.Attributes()("foo", true)("bar", false);
        UNIT_ASSERT(node.HasAttributes());

        {
            TNode copyNode;
            UNIT_ASSERT(!copyNode.HasAttributes());
            copyNode = node;
            UNIT_ASSERT(copyNode.HasAttributes());
            UNIT_ASSERT_EQUAL(copyNode.GetAttributes()["foo"], TNode(true));
        }

        {
            TNode movedWithoutAttributes(42);
            movedWithoutAttributes.Attributes()("one", 1)("two", 2);
            movedWithoutAttributes.MoveWithoutAttributes(TNode(node));
            UNIT_ASSERT(movedWithoutAttributes.IsMap());
            UNIT_ASSERT_EQUAL(movedWithoutAttributes["lee"], TNode(42));
            UNIT_ASSERT_EQUAL(movedWithoutAttributes.GetAttributes()["one"], TNode(1));
            UNIT_ASSERT(!movedWithoutAttributes.GetAttributes().HasKey("foo"));
        }

        {
            TNode copyNode = node;
            UNIT_ASSERT(copyNode.HasAttributes());
            UNIT_ASSERT(copyNode.GetAttributes().HasKey("foo"));
            copyNode.ClearAttributes();
            UNIT_ASSERT(!copyNode.HasAttributes());
            UNIT_ASSERT(!copyNode.GetAttributes().HasKey("foo"));
        }

        {
            TNode copyNode = node;
            UNIT_ASSERT(copyNode.HasAttributes());
            UNIT_ASSERT(copyNode.GetAttributes().HasKey("foo"));
            copyNode.Clear();
            UNIT_ASSERT(!copyNode.HasAttributes());
            UNIT_ASSERT(!copyNode.GetAttributes().HasKey("foo"));
        }
    }

    Y_UNIT_TEST(TestEq) {
        TNode nodeNoAttributes = TNode()("lee", 42)("faa", 54);
        TNode node = nodeNoAttributes;
        node.Attributes()("foo", true)("bar", false);
        UNIT_ASSERT(node != nodeNoAttributes);
        UNIT_ASSERT(nodeNoAttributes != node);
        TNode copyNode = node;
        UNIT_ASSERT(copyNode == node);
        UNIT_ASSERT(node == copyNode);
    }

    Y_UNIT_TEST(TestComparison) {
        using namespace NYT::NNodeCmp;
        {
            TNode nodeNoAttributes = TNode()("lee", 42)("faa", 54);
            TNode node = nodeNoAttributes;
            node.Attributes()("foo", true)("bar", false);
            UNIT_ASSERT_EXCEPTION(node > nodeNoAttributes, TNode::TTypeError);
            UNIT_ASSERT_EXCEPTION(node >= nodeNoAttributes, TNode::TTypeError);
            UNIT_ASSERT_EXCEPTION(nodeNoAttributes < node, TNode::TTypeError);
            UNIT_ASSERT_EXCEPTION(nodeNoAttributes <= node, TNode::TTypeError);
        }
        {
            TNode nodeMap = TNode()("map", 23);
            TNode nodeList = TNode::CreateList();
            UNIT_ASSERT_EXCEPTION(nodeList > nodeMap, TNode::TTypeError);
            UNIT_ASSERT_EXCEPTION(nodeMap < nodeList, TNode::TTypeError);
            UNIT_ASSERT_EXCEPTION(nodeMap >= nodeMap, TNode::TTypeError);
            UNIT_ASSERT_EXCEPTION(nodeList <= nodeList, TNode::TTypeError);
        }
        {
            TNode node1("aaa");
            TNode node2("bbb");
            TNode node3("ccc");
            UNIT_ASSERT(node1 < node2);
            UNIT_ASSERT(node1 <= node2);
            UNIT_ASSERT(node1 < node3);
            UNIT_ASSERT(node1 <= node3);
            UNIT_ASSERT(!(node3 < node1));
            UNIT_ASSERT(!(node1 > node3));
            UNIT_ASSERT(!(node3 <= node1));
            UNIT_ASSERT(!(node1 >= node3));

            UNIT_ASSERT(node3 > node2);
            UNIT_ASSERT(node3 >= node2);
            UNIT_ASSERT(node3 > node1);
            UNIT_ASSERT(node3 >= node1);

            UNIT_ASSERT(node1 <= node1);
            UNIT_ASSERT(node1 >= node1);
        }
        {
            TNode node1(23);
            TNode node2("bbb");
            TNode node3 = TNode::CreateEntity();

            UNIT_ASSERT(node1 > node2);
            UNIT_ASSERT(node1 >= node2);
            UNIT_ASSERT(node2 < node1);
            UNIT_ASSERT(node2 <= node1);

            UNIT_ASSERT(!(node1 < node2));
            UNIT_ASSERT(!(node1 <= node2));
            UNIT_ASSERT(!(node2 > node1));
            UNIT_ASSERT(!(node2 >= node1));

            UNIT_ASSERT(node1 < node3);
            UNIT_ASSERT(node2 < node3);
            UNIT_ASSERT(node3 <= node3);
            UNIT_ASSERT(!(node3 < node3));
            UNIT_ASSERT(!(node3 > node3));
            UNIT_ASSERT(!(node2 >= node3));
        }
    }

    Y_UNIT_TEST(TestSaveLoad) {
        TNode node = TNode()("foo", "bar")("baz", 42);
        node.Attributes()["attr_name"] = "attr_value";

        TString bytes;
        {
            TStringOutput s(bytes);
            ::Save(&s, node);
        }

        TNode nodeCopy;
        {
            TStringInput s(bytes);
            ::Load(&s, nodeCopy);
        }

        UNIT_ASSERT_VALUES_EQUAL(node, nodeCopy);
    }

    Y_UNIT_TEST(TestIntCast) {
        TNode node = 1ull << 31;
        UNIT_ASSERT(node.IsUint64());
        UNIT_ASSERT_EXCEPTION(node.IntCast<i32>(), TNode::TTypeError);
        UNIT_ASSERT(node.IntCast<ui32>() == static_cast<ui32>(node.AsUint64()));
        UNIT_ASSERT(node.IntCast<i64>() == static_cast<i64>(node.AsUint64()));
        UNIT_ASSERT(node.IntCast<ui64>() == node.AsUint64());

        node = 1ull << 63;
        UNIT_ASSERT(node.IsUint64());
        UNIT_ASSERT_EXCEPTION(node.IntCast<i64>(), TNode::TTypeError);
        UNIT_ASSERT(node.IntCast<ui64>() == node.AsUint64());

        node = 12345;
        UNIT_ASSERT(node.IsInt64());
        UNIT_ASSERT_EXCEPTION(node.IntCast<i8>(), TNode::TTypeError);
        UNIT_ASSERT_EXCEPTION(node.IntCast<ui8>(), TNode::TTypeError);
        UNIT_ASSERT(node.IntCast<i16>() == static_cast<i16>(node.AsInt64()));
        UNIT_ASSERT(node.IntCast<ui16>() == static_cast<ui16>(node.AsInt64()));
        UNIT_ASSERT(node.IntCast<i32>() == static_cast<i32>(node.AsInt64()));
        UNIT_ASSERT(node.IntCast<ui32>() == static_cast<ui32>(node.AsInt64()));
        UNIT_ASSERT(node.IntCast<i64>() == node.AsInt64());
        UNIT_ASSERT(node.IntCast<ui64>() == static_cast<ui64>(node.AsInt64()));

        node = -5;
        UNIT_ASSERT(node.IsInt64());
        UNIT_ASSERT(node.IntCast<i8>() == static_cast<i8>(node.AsInt64()));
        UNIT_ASSERT(node.IntCast<i16>() == static_cast<i16>(node.AsInt64()));
        UNIT_ASSERT(node.IntCast<i32>() == static_cast<i32>(node.AsInt64()));
        UNIT_ASSERT(node.IntCast<i64>() == node.AsInt64());
        UNIT_ASSERT_EXCEPTION(node.IntCast<ui8>(), TNode::TTypeError);
        UNIT_ASSERT_EXCEPTION(node.IntCast<ui16>(), TNode::TTypeError);
        UNIT_ASSERT_EXCEPTION(node.IntCast<ui32>(), TNode::TTypeError);
        UNIT_ASSERT_EXCEPTION(node.IntCast<ui64>(), TNode::TTypeError);
    }

    Y_UNIT_TEST(TestConvertToString) {
        UNIT_ASSERT_VALUES_EQUAL(TNode(5).ConvertTo<TString>(), "5");
        UNIT_ASSERT_VALUES_EQUAL(TNode(123432423).ConvertTo<TString>(), "123432423");
        UNIT_ASSERT_VALUES_EQUAL(TNode(123456789012345678ll).ConvertTo<TString>(), "123456789012345678");
        UNIT_ASSERT_VALUES_EQUAL(TNode(123456789012345678ull).ConvertTo<TString>(), "123456789012345678");
        UNIT_ASSERT_VALUES_EQUAL(TNode(-123456789012345678ll).ConvertTo<TString>(), "-123456789012345678");
        UNIT_ASSERT_VALUES_EQUAL(TNode(true).ConvertTo<TString>(), "1");
        UNIT_ASSERT_VALUES_EQUAL(TNode(false).ConvertTo<TString>(), "0");
        UNIT_ASSERT_VALUES_EQUAL(TNode(5.3).ConvertTo<TString>(), "5.3");
    }

    Y_UNIT_TEST(TestConvertFromString) {
        UNIT_ASSERT_VALUES_EQUAL(TNode("123456789012345678").ConvertTo<ui64>(), 123456789012345678ull);
        UNIT_ASSERT_VALUES_EQUAL(TNode("123456789012345678").ConvertTo<i64>(), 123456789012345678);
        UNIT_ASSERT_VALUES_EQUAL(TNode(ToString(1ull << 63)).ConvertTo<ui64>(), 1ull << 63);
        UNIT_ASSERT_EXCEPTION(TNode(ToString(1ull << 63)).ConvertTo<i64>(), TFromStringException);
        UNIT_ASSERT_VALUES_EQUAL(TNode("5.34").ConvertTo<double>(), 5.34);
    }

    Y_UNIT_TEST(TestConvertDoubleInt) {
        UNIT_ASSERT_VALUES_EQUAL(TNode(5.3).ConvertTo<i8>(), 5);
        UNIT_ASSERT_VALUES_EQUAL(TNode(5.3).ConvertTo<ui8>(), 5);
        UNIT_ASSERT_VALUES_EQUAL(TNode(5.3).ConvertTo<i64>(), 5);
        UNIT_ASSERT_VALUES_EQUAL(TNode(5.3).ConvertTo<ui64>(), 5);

        UNIT_ASSERT_VALUES_EQUAL(TNode(-5.3).ConvertTo<i8>(), -5);
        UNIT_ASSERT_VALUES_EQUAL(TNode(-5.3).ConvertTo<i64>(), -5);
        UNIT_ASSERT_EXCEPTION(TNode(-5.3).ConvertTo<ui8>(), TNode::TTypeError);
        UNIT_ASSERT_EXCEPTION(TNode(-5.3).ConvertTo<ui64>(), TNode::TTypeError);

        UNIT_ASSERT_EXCEPTION(TNode(256.0).ConvertTo<i8>(), TNode::TTypeError);
        UNIT_ASSERT_EXCEPTION(TNode(256.0).ConvertTo<ui8>(), TNode::TTypeError);
        UNIT_ASSERT_EXCEPTION(TNode(1e100).ConvertTo<i64>(), TNode::TTypeError);
        UNIT_ASSERT_EXCEPTION(TNode(1e100).ConvertTo<ui64>(), TNode::TTypeError);
        {
            double v = (1ull << 63) + (1ull);
            TNode node = v;
            UNIT_ASSERT(node.IsDouble());
            UNIT_ASSERT_EXCEPTION(node.ConvertTo<i64>(), TNode::TTypeError);
            UNIT_ASSERT_VALUES_EQUAL(node.ConvertTo<ui64>(), static_cast<ui64>(v));
        }
        {
            double v = (double)(1ull << 63) + (1ull << 63);
            TNode node = v;
            UNIT_ASSERT(node.IsDouble());
            UNIT_ASSERT_EXCEPTION(node.ConvertTo<i64>(), TNode::TTypeError);
            UNIT_ASSERT_EXCEPTION(node.ConvertTo<ui64>(), TNode::TTypeError);
        }
        UNIT_ASSERT_EXCEPTION(TNode(NAN).ConvertTo<ui64>(), TNode::TTypeError);
        UNIT_ASSERT_EXCEPTION(TNode(NAN).ConvertTo<i64>(), TNode::TTypeError);

        UNIT_ASSERT_EXCEPTION(TNode(INFINITY).ConvertTo<ui64>(), TNode::TTypeError);
        UNIT_ASSERT_EXCEPTION(TNode(INFINITY).ConvertTo<i64>(), TNode::TTypeError);
    }

    Y_UNIT_TEST(TestConvertToBool) {
        UNIT_ASSERT_VALUES_EQUAL(TNode("true").ConvertTo<bool>(), true);
        UNIT_ASSERT_VALUES_EQUAL(TNode("TRUE").ConvertTo<bool>(), true);
        UNIT_ASSERT_VALUES_EQUAL(TNode("false").ConvertTo<bool>(), false);
        UNIT_ASSERT_VALUES_EQUAL(TNode("FALSE").ConvertTo<bool>(), false);
        UNIT_ASSERT_VALUES_EQUAL(TNode(1).ConvertTo<bool>(), true);
        UNIT_ASSERT_VALUES_EQUAL(TNode(0).ConvertTo<bool>(), false);
        UNIT_ASSERT_EXCEPTION(TNode("random").ConvertTo<bool>(), TFromStringException);
        UNIT_ASSERT_EXCEPTION(TNode("").ConvertTo<bool>(), TFromStringException);
    }

    Y_UNIT_TEST(TestCanonicalSerialization) {
        auto node = TNode()
            ("ca", "ca")("c", "c")("a", "a")("b", "b")
            ("bb", TNode()
                ("ii", "ii")("i", "i")("jj", "jj"));
        node.Attributes() = TNode()("za", "za")("z", "z")("xxx", "xxx")("xx", "xx");
        UNIT_ASSERT_VALUES_EQUAL(NodeToCanonicalYsonString(node),
            "<\"xx\"=\"xx\";\"xxx\"=\"xxx\";\"z\"=\"z\";\"za\"=\"za\">"
            "{\"a\"=\"a\";\"b\"=\"b\";\"bb\"="
                "{\"i\"=\"i\";\"ii\"=\"ii\";\"jj\"=\"jj\"};"
            "\"c\"=\"c\";\"ca\"=\"ca\"}");
    }

    Y_UNIT_TEST(OperatorEqualSubnode) {
        TNode node;
        node["a"]["b"] = "c";

        node = node["a"];
        node = node["b"];

        UNIT_ASSERT_VALUES_EQUAL(node.AsString(), "c");
    }

    Y_UNIT_TEST(TestMapGetters) {
        auto node = TNode::CreateMap()
            ("string", "7")
            ("int64", 3)
            ("uint64", 5u)
            ("double", -3.5)
            ("list", TNode::CreateList().Add(5))
            ("map", TNode::CreateMap()("key", "value"));

        UNIT_ASSERT_VALUES_EQUAL(node.ChildAs<TString>("string"), "7");
        UNIT_ASSERT_VALUES_EQUAL(node.ChildAsString("string"), "7");
        UNIT_ASSERT_VALUES_EQUAL(node.ChildConvertTo<i64>("string"), 7);

        UNIT_ASSERT_VALUES_EQUAL(node.ChildAs<i64>("int64"), 3);
        UNIT_ASSERT_VALUES_EQUAL(node.ChildAsInt64("int64"), 3);
        UNIT_ASSERT_VALUES_EQUAL(node.ChildIntCast<ui64>("int64"), 3u);

        UNIT_ASSERT_VALUES_EQUAL(node.ChildAs<ui64>("uint64"), 5u);
        UNIT_ASSERT_VALUES_EQUAL(node.ChildAsUint64("uint64"), 5u);
        UNIT_ASSERT_VALUES_EQUAL(node.ChildIntCast<i64>("uint64"), 5);
        UNIT_ASSERT_VALUES_EQUAL(node.ChildConvertTo<TString>("uint64"), "5");

        UNIT_ASSERT_VALUES_EQUAL(node.ChildAs<double>("double"), -3.5);
        UNIT_ASSERT_VALUES_EQUAL(node.ChildAsDouble("double"), -3.5);
        UNIT_ASSERT_VALUES_EQUAL(node.ChildConvertTo<TString>("double"), "-3.5");

        UNIT_ASSERT_VALUES_EQUAL(node.ChildAs<TNode::TListType>("list")[0].AsInt64(), 5);
        UNIT_ASSERT_VALUES_EQUAL(node.ChildAsList("list")[0].AsInt64(), 5);

        UNIT_ASSERT_VALUES_EQUAL(node.ChildAs<TNode::TMapType>("map")["key"].AsString(), "value");
        UNIT_ASSERT_VALUES_EQUAL(node.ChildAsMap("map")["key"].AsString(), "value");

        // mutable accessor
        auto& childString = node.ChildAs<TString>("string");
        childString = "yaddayadda";
        UNIT_ASSERT_VALUES_EQUAL(node.ChildAs<TString>("string"), "yaddayadda");
    }

    Y_UNIT_TEST(TestListGetters) {
        auto node = TNode::CreateList()
            .Add("7")
            .Add(3)
            .Add(5u)
            .Add(-3.5)
            .Add(TNode::CreateList().Add(5))
            .Add(TNode::CreateMap()("key", "value"));

        UNIT_ASSERT_VALUES_EQUAL(node.ChildAs<TString>(0), "7");
        UNIT_ASSERT_VALUES_EQUAL(node.ChildAsString(0), "7");
        UNIT_ASSERT_VALUES_EQUAL(node.ChildConvertTo<i64>(0), 7);

        UNIT_ASSERT_VALUES_EQUAL(node.ChildAs<i64>(1), 3);
        UNIT_ASSERT_VALUES_EQUAL(node.ChildAsInt64(1), 3);
        UNIT_ASSERT_VALUES_EQUAL(node.ChildIntCast<ui64>(1), 3u);

        UNIT_ASSERT_VALUES_EQUAL(node.ChildAs<ui64>(2), 5u);
        UNIT_ASSERT_VALUES_EQUAL(node.ChildAsUint64(2), 5u);
        UNIT_ASSERT_VALUES_EQUAL(node.ChildIntCast<i64>(2), 5);
        UNIT_ASSERT_VALUES_EQUAL(node.ChildConvertTo<TString>(2), "5");

        UNIT_ASSERT_VALUES_EQUAL(node.ChildAs<double>(3), -3.5);
        UNIT_ASSERT_VALUES_EQUAL(node.ChildAsDouble(3), -3.5);
        UNIT_ASSERT_VALUES_EQUAL(node.ChildConvertTo<TString>(3), "-3.5");

        UNIT_ASSERT_VALUES_EQUAL(node.ChildAs<TNode::TListType>(4)[0].AsInt64(), 5);
        UNIT_ASSERT_VALUES_EQUAL(node.ChildAsList(4)[0].AsInt64(), 5);

        UNIT_ASSERT_VALUES_EQUAL(node.ChildAs<TNode::TMapType>(5)["key"].AsString(), "value");
        UNIT_ASSERT_VALUES_EQUAL(node.ChildAsMap(5)["key"].AsString(), "value");

        // mutable accessor
        auto& childString = node.ChildAs<TString>(0);
        childString = "yaddayadda";
        UNIT_ASSERT_VALUES_EQUAL(node.ChildAs<TString>(0), "yaddayadda");
    }
}
