#include <library/cpp/testing/gtest/gtest.h>

#include <library/cpp/yt/misc/preprocessor.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TPreprocessorTest, Concatenation)
{
    EXPECT_EQ(12, PP_CONCAT(1, 2));
    EXPECT_STREQ("FooBar", PP_STRINGIZE(PP_CONCAT(Foo, Bar)));
}

TEST(TPreprocessorTest, Stringize)
{
    EXPECT_STREQ(PP_STRINGIZE(123456), "123456");
    EXPECT_STREQ(PP_STRINGIZE(FooBar), "FooBar");
    EXPECT_STREQ(PP_STRINGIZE(T::XYZ), "T::XYZ");
}

TEST(TPreprocessorTest, Count)
{
    EXPECT_EQ(0, PP_COUNT());
    EXPECT_EQ(1, PP_COUNT((0)));
    EXPECT_EQ(2, PP_COUNT((0)(0)));
    EXPECT_EQ(3, PP_COUNT((0)(0)(0)));
    EXPECT_EQ(4, PP_COUNT((0)(0)(0)(0)));
    EXPECT_EQ(5, PP_COUNT((0)(0)(0)(0)(0)));
}

TEST(TPreprocessorTest, Kill)
{
    EXPECT_STREQ("PP_NIL (0)",       PP_STRINGIZE(PP_NIL PP_KILL((0), 0)));
    EXPECT_STREQ("PP_NIL",           PP_STRINGIZE(PP_NIL PP_KILL((0), 1)));
    EXPECT_STREQ("PP_NIL (0)(1)(2)", PP_STRINGIZE(PP_NIL PP_KILL((0)(1)(2), 0)));
    EXPECT_STREQ("PP_NIL (1)(2)",    PP_STRINGIZE(PP_NIL PP_KILL((0)(1)(2), 1)));
    EXPECT_STREQ("PP_NIL (2)",       PP_STRINGIZE(PP_NIL PP_KILL((0)(1)(2), 2)));
    EXPECT_STREQ("PP_NIL",           PP_STRINGIZE(PP_NIL PP_KILL((0)(1)(2), 3)));
}

TEST(TPreprocessorTest, Head)
{
    EXPECT_STREQ("0", PP_STRINGIZE(PP_HEAD((0))));
    EXPECT_STREQ("0", PP_STRINGIZE(PP_HEAD((0)(1))));
    EXPECT_STREQ("0", PP_STRINGIZE(PP_HEAD((0)(1)(2))));
}

TEST(TPreprocessorTest, Tail)
{
    EXPECT_STREQ("PP_NIL",        PP_STRINGIZE(PP_NIL PP_TAIL((0))));
    EXPECT_STREQ("PP_NIL (1)",    PP_STRINGIZE(PP_NIL PP_TAIL((0)(1))));
    EXPECT_STREQ("PP_NIL (1)(2)", PP_STRINGIZE(PP_NIL PP_TAIL((0)(1)(2))));
}

TEST(TPreprocessorTest, Range)
{
    EXPECT_EQ(0, PP_HEAD(PP_RANGE(0, 0)));
    EXPECT_EQ(1, PP_COUNT(PP_RANGE(0, 0)));
    EXPECT_EQ(1, PP_COUNT(PP_RANGE(8, 8)));
    EXPECT_EQ(8, PP_HEAD(PP_RANGE(8, 8)));

    EXPECT_EQ(8, PP_COUNT(PP_RANGE(1, 8)));
    EXPECT_EQ(1, PP_ELEMENT(PP_RANGE(1, 8), 0));
    EXPECT_EQ(4, PP_ELEMENT(PP_RANGE(1, 8), 3));
    EXPECT_EQ(8, PP_ELEMENT(PP_RANGE(1, 8), 7));

    EXPECT_EQ(300, PP_COUNT(PP_RANGE(0, 299)));
    EXPECT_EQ(300, PP_COUNT(PP_RANGE(1, 300)));
    EXPECT_EQ(300, PP_ELEMENT(PP_RANGE(1, 300), 299));
    EXPECT_EQ(1, PP_COUNT(PP_RANGE(300, 300)));
    EXPECT_EQ(300, PP_HEAD(PP_RANGE(300, 300)));

#define TEST_RANGE_FIRST 2
#define TEST_RANGE_LAST 5
    EXPECT_EQ(4, PP_COUNT(PP_RANGE(TEST_RANGE_FIRST, TEST_RANGE_LAST)));
    EXPECT_EQ(2, PP_HEAD(PP_RANGE(TEST_RANGE_FIRST, TEST_RANGE_LAST)));
    EXPECT_EQ(5, PP_ELEMENT(PP_RANGE(TEST_RANGE_FIRST, TEST_RANGE_LAST), 3));
#undef TEST_RANGE_FIRST
#undef TEST_RANGE_LAST

#define ADD_RANGE_VALUE(value) + value
    EXPECT_EQ(36, (0 PP_FOR_EACH(ADD_RANGE_VALUE, PP_RANGE(1, 8))));
    EXPECT_EQ(7, (0 PP_FOR_EACH(ADD_RANGE_VALUE, PP_RANGE(7, 7))));
    EXPECT_EQ(45150, (0 PP_FOR_EACH(ADD_RANGE_VALUE, PP_RANGE(1, 300))));
#undef ADD_RANGE_VALUE
}

TEST(TPreprocessorTest, ForEach)
{
    EXPECT_STREQ(
        "\"Foo\" \"Bar\" \"Spam\" \"Ham\"",
        PP_STRINGIZE(PP_FOR_EACH(PP_STRINGIZE, (Foo)(Bar)(Spam)(Ham)))
    );
#define my_functor(x) +x+
    EXPECT_STREQ(
        "+1+ +2+ +3+",
        PP_STRINGIZE(PP_FOR_EACH(my_functor, (1)(2)(3)))
    );
#undef  my_functor
}

TEST(TPreprocessorTest, MakeSingleton)
{
    EXPECT_EQ(1, PP_ELEMENT((1), 0));
    EXPECT_EQ(1, PP_ELEMENT((1)(2), 0));
    EXPECT_EQ(2, PP_ELEMENT((1)(2), 1));
    EXPECT_EQ(1, PP_ELEMENT((1)(2)(3), 0));
    EXPECT_EQ(2, PP_ELEMENT((1)(2)(3), 1));
    EXPECT_EQ(3, PP_ELEMENT((1)(2)(3), 2));
    EXPECT_EQ(1, PP_ELEMENT((1)(2)(3)(4), 0));
    EXPECT_EQ(2, PP_ELEMENT((1)(2)(3)(4), 1));
    EXPECT_EQ(3, PP_ELEMENT((1)(2)(3)(4), 2));
    EXPECT_EQ(4, PP_ELEMENT((1)(2)(3)(4), 3));
}

TEST(TPreprocessorTest, Conditional)
{
    EXPECT_EQ(1, PP_IF(PP_TRUE,  1, 2));
    EXPECT_EQ(2, PP_IF(PP_FALSE, 1, 2));
}

TEST(TPreprocessorTest, IsSequence)
{
    EXPECT_STREQ("PP_FALSE", PP_STRINGIZE(PP_IS_SEQUENCE( 0    )));
    EXPECT_STREQ("PP_TRUE",  PP_STRINGIZE(PP_IS_SEQUENCE((0)   )));
    EXPECT_STREQ("PP_TRUE",  PP_STRINGIZE(PP_IS_SEQUENCE((0)(0))));
    EXPECT_STREQ("PP_FALSE", PP_STRINGIZE(PP_IS_SEQUENCE(PP_NIL)));
}

TEST(TPreprocessorTest, Deparen)
{
    EXPECT_STREQ("a", PP_STRINGIZE(PP_DEPAREN(a)));
    EXPECT_STREQ("a", PP_STRINGIZE(PP_DEPAREN((a))));
    EXPECT_STREQ("( a, b)", PP_STRINGIZE((PP_DEPAREN(a, b))));
    EXPECT_STREQ("( a, b)", PP_STRINGIZE((PP_DEPAREN((a, b)))));
}

TEST(TPreprocessorTest, IsEmpty)
{
    EXPECT_EQ(1, PP_IF(PP_IS_EMPTY(), 1, 2));
    EXPECT_EQ(1, PP_IF(PP_IS_EMPTY(PP_EMPTINESS), 1, 2));
    EXPECT_EQ(1, PP_IF(PP_IS_EMPTY(PP_EMPTY()), 1, 2));
    EXPECT_EQ(1, PP_IF(PP_IS_EMPTY(PP_EMPTY PP_LEFT_PARENTHESIS PP_RIGHT_PARENTHESIS), 1, 2));

    EXPECT_EQ(2, PP_IF(PP_IS_EMPTY(()), 1, 2));
    EXPECT_EQ(2, PP_IF(PP_IS_EMPTY(PP_EMPTY), 1, 2));
    EXPECT_EQ(2, PP_IF(PP_IS_EMPTY(0), 1, 2));
    EXPECT_EQ(2, PP_IF(PP_IS_EMPTY(hello), 1, 2));
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
