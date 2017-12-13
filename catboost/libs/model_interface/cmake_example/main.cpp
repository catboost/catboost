#include "../wrapped_calcer.h"
#include <iostream>

int main(int argc, char** argv) {
    {
        ModelCalcerWrapper calcer;
        calcer.init_from_file("model.bin");
        std::cout << calcer.CalcFlat(std::vector<float>(100)) << std::endl;
        std::cout << calcer.CalcFlat(std::vector<float>(100, 1.0f)) << std::endl;
    }
    {
        ModelCalcerWrapper calcer("model.bin");
        std::cout << calcer.CalcFlat(std::vector<float>(100)) << std::endl;
        std::cout << calcer.CalcFlat(std::vector<float>(100, 1.0f)) << std::endl;
    }
    return 0;
}

