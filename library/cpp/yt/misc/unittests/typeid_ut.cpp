#include "typeid_sample.h"

#include <library/cpp/yt/misc/typeid.h>

#include <library/cpp/testing/gtest/gtest.h>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TTypeidTest, Complete)
{
    EXPECT_NE(std::string(Typeid<TTypeidComplete>().name()).find("TTypeidComplete"), std::string::npos);
}

TEST(TTypeidTest, Incomplete)
{
    EXPECT_NE(std::string(Typeid<TTypeidIncomplete>().name()).find("TTypeidIncomplete"), std::string::npos);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
