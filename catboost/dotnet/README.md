CatBoost .NET Inference API
===

This is the root directory of the CatBoost .NET API codebase. From here, you can:

* hack around with the model API — navigate to `CatBoostNet\README.md`;
* run unit tests — execute `dotnet test` from this directory or navigate to the `CatBoostNetTests` project;
* take a look at the usage example — navigate to `HeartDiseaseDemo\README.md`;

Or, if you just want to explore the whole codebase, feel free to open `dotnet.sln` and explore on your own ;)

Requirements
---

* Windows 10 or 11 (so far — ideally, the build instructions should be ported to the `linux` / `osx` runtimes without too much effort);
* .NET Core 3.1 CLI;
* [Build environment setup for CMake](https://catboost.ai/docs/en/installation/build-environment-setup-for-cmake), see instructions for building without CUDA support (if you need to compile the API `.nupkg` by yourself)
