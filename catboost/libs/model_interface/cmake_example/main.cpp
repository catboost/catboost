#include "../wrapped_calcer.h"
#include <iostream>

int main(int argc, char** argv) {
    ModelCalcerWrapper calcer("model.bin");
    std::cout << calcer.Calc(std::vector<float>(100)) << std::endl;
    std::cout << calcer.Calc(std::vector<float>(100, 1.0f)) << std::endl;
    return 0;
}