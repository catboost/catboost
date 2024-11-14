#include <array>
           struct ExtensionDescriptor {
              const char* Name;
              const char* RequiredLibraries[1 + 1 + 24];
           };
           std::array<ExtensionDescriptor, 1> AvailableExtensions{
ExtensionDescriptor{"Polly", {"support", "core", "scalaropts", "instcombine", "transformutils", "analysis", "ipo", "mc", "passes", "linker", "irreader", "analysis", "bitreader", "mcparser", "object", "profiledata", "target", "targetparser", "vectorize", "nvptxcodegen", "nvptxdesc", "nvptxinfo", "PollyISL", "PollyPPCG", "Polly",nullptr}},
};
