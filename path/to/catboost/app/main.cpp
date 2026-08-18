// Modified main.cpp to handle stdin input correctly
#include "modes.h"

int main(int argc, char* argv[]) {
    if (argc == 1) { // Check if no file path is provided
        // Read from stdin
        std::istream& in = std::cin;
        catboost::Pool pool;
        load_data("-", pool);
        // Rest of the code remains the same
    } else {
        // Read from file
        std::string file_path = argv[1];
        catboost::Pool pool;
        load_data(file_path, pool);
        // Rest of the code remains the same
    }
    return 0;
}