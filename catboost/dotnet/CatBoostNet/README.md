CatBoost .NET Inference API library
===

This project contains definition of the inference API. Somewhere in the future you will be able to find the `.nupkg` containing this library in the NuGet Gallery — but so far, you can consume this library in two ways:

Import the project directly into your solution.
---

Just copy this directory to whatever solution you need — or start hacking right here, in this solution. Keep in mind that to use this project in your own solution you'll also need to:

* copy `LibraryBuilder` project to the solution as well;
* fix build instructions in the `LibraryBuilder` — in this repository these instructions assume that the library is compiled in the `catboost` repository.

Compile stand-alone `.nupkg`
---

Here you'll also need [NuGet CLI](https://www.nuget.org/downloads) — download it (it's a standalone executable) and add it in your `PATH`. To build actually usable `.nupkg`:

* build this project using Visual Studio 2022;
* pack this project — you should find `CatBoostNet.0.1.1.nupkg` in the same directory with the build artifacts (e.g., `bin\Debug`);
* choose a directory (`<PATH>`) and copy `CatBoostNet.0.1.1.nupkg` there;
* add a new NuGet source: (in PowerShell): `nuget sources add -Name <YOURNAME> -Source <PATH>`. After this, you should be able to find `CatBoostNet` in the NuGet Extension Manager in Visual Studio — just don't forget to switch the package source to `<YOURNAME>`.
