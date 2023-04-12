CatBoost .NET Inference API
===

This is the root directory of the CatBoost .NET API codebase. From here, you can:

* hack around with the model API — navigate to `CatBoostNet\README.md`;
* run unit tests — execute `dotnet test` from this directory or navigate to the `CatBoostNetTests` project;
* take a look at the usage example — navigate to `HeartDiseaseDemo\README.md`;

Or, if you just want to explore the whole codebase, feel free to open `dotnet.sln` and explore on your own ;)

Requirements
---

* Win10 (so far — ideally, the build instructions should be ported to the `linux` / `osx` runtimes without too much effort);
* .NET Core 3.1 CLI;
* Visual Studio 2019 (if you need to compile the API `.nupkg` by yourself)

This project also assumes that it can access `ya make` — if you cloned the complete `catboost` repository, you should be fine :)
