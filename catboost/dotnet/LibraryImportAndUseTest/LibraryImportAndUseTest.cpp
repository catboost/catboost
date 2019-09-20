// LibraryImportAndUseTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"


int main(int argc, const char** argv)
{
    if (argc != 2) {
        std::cout << "Please, provide path to test model" << std::endl;
        return 1;
    }
    std::string modelPath(argv[1]);
    std::cout << "started" << std::endl;
    {
        std::vector<std::vector<float>> floatFeatures;
        std::vector<std::vector<std::string>> catFeatures;
        {
            ModelCalcerWrapper wrapper(modelPath);

            std::cout << wrapper.GetTreeCount() << std::endl;
            std::cout << wrapper.GetFloatFeaturesCount() << std::endl;
            std::cout << wrapper.GetCatFeaturesCount() << std::endl;
            const size_t pseudodocs = 1000;
            const auto floatFeaturesCount = wrapper.GetFloatFeaturesCount();
            const auto catFeaturesCount = wrapper.GetCatFeaturesCount();
            floatFeatures = std::vector<std::vector<float>>(pseudodocs, std::vector<float>(floatFeaturesCount, 0.0f));
            catFeatures = std::vector<std::vector<std::string>>(pseudodocs, std::vector<std::string>(catFeaturesCount, "a"));
        }

        for (size_t i = 0; i < 50; ++i) {
            ModelCalcerWrapper wrapper(modelPath);
            wrapper.Calc(floatFeatures, catFeatures);
        }
    }
    std::cout << "finished" << std::endl;
    return 0;
}
