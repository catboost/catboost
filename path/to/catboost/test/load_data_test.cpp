// Added a unit test for the modified load_data function
#include "gtest/gtest.h"
#include "load_data.cpp"

TEST(LoadDataTest, StdinInput) {
    std::istringstream in("1 2 3\n4 5 6");
    catboost::Pool pool;
    load_data("-", pool);
    EXPECT_EQ(pool.size(), 2);
}

TEST(LoadDataTest, FileInput) {
    std::string file_path = "test_data.csv";
    std::ofstream out(file_path);
    out << "1 2 3\n4 5 6";
    out.close();
    catboost::Pool pool;
    load_data(file_path, pool);
    EXPECT_EQ(pool.size(), 2);
}