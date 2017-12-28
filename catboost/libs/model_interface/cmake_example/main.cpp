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
        std::vector<std::string> catFeatures = {"1", "2", "3"};
        std::cout << calcer.Calc(std::vector<float>(100), catFeatures) << std::endl;
        std::cout << calcer.Calc(std::vector<float>(100, 1.0f), std::vector<std::string>()) << std::endl;
    }
    return 0;
}

