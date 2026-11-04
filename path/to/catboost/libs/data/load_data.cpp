// Modified load_data function to handle stdin input correctly
#include "data_reader.h"

void load_data(const std::string& file_path, catboost::Pool& pool) {
    if (file_path == "-") { // Check if file path is stdin
        // Read from stdin
        std::istream& in = std::cin;
        catboost::DataStreamReader reader(in);
        reader.read(pool);
    } else {
        // Read from file
        std::ifstream in(file_path);
        if (!in.is_open()) {
            throw TCatboostException("Failed to open file: " + file_path);
        }
        catboost::DataStreamReader reader(in);
        reader.read(pool);
    }
}