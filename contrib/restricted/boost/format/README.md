Format, part of the collection of [Boost C++ Libraries](https://github.com/boostorg), provides a type-safe mechanism for formatting arguments according to a printf-like format-string.  User-defined types are supported by providing a `std::ostream operator <<` implementation for them.

### License

Distributed under the [Boost Software License, Version 1.0](https://www.boost.org/LICENSE_1_0.txt).

### Properties

* C++11
* Header-only

### Build Status

<!-- boost-ci/tools/makebadges.sh --project format --appveyor aeg8obnkb0mrgqvd --codecov  --coverity 14007 -->
| Branch          | GHA CI | Appveyor | Coverity Scan | codecov.io | Deps | Docs | Tests |
| :-------------: | ------ | -------- | ------------- | ---------- | ---- | ---- | ----- |
| [`master`](https://github.com/boostorg/format/tree/master) | [![Build Status](https://github.com/boostorg/format/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/boostorg/format/actions?query=branch:master) | [![Build status](https://ci.appveyor.com/api/projects/status/aeg8obnkb0mrgqvd/branch/master?svg=true)](https://ci.appveyor.com/project/cppalliance/format/branch/master) | [![Coverity Scan Build Status](https://scan.coverity.com/projects/14007/badge.svg)](https://scan.coverity.com/projects/boostorg-format) | [![codecov](https://codecov.io/gh/boostorg/format/branch/master/graph/badge.svg?token=)](https://codecov.io/gh/boostorg/format/tree/master) | [![Deps](https://img.shields.io/badge/deps-master-brightgreen.svg)](https://pdimov.github.io/boostdep-report/master/format.html) | [![Documentation](https://img.shields.io/badge/docs-master-brightgreen.svg)](https://www.boost.org/doc/libs/master/libs/format) | [![Enter the Matrix](https://img.shields.io/badge/matrix-master-brightgreen.svg)](https://www.boost.org/development/tests/master/developer/format.html)
| [`develop`](https://github.com/boostorg/format/tree/develop) | [![Build Status](https://github.com/boostorg/format/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/boostorg/format/actions?query=branch:develop) | [![Build status](https://ci.appveyor.com/api/projects/status/aeg8obnkb0mrgqvd/branch/develop?svg=true)](https://ci.appveyor.com/project/cppalliance/format/branch/develop) | [![Coverity Scan Build Status](https://scan.coverity.com/projects/14007/badge.svg)](https://scan.coverity.com/projects/boostorg-format) | [![codecov](https://codecov.io/gh/boostorg/format/branch/develop/graph/badge.svg?token=)](https://codecov.io/gh/boostorg/format/tree/develop) | [![Deps](https://img.shields.io/badge/deps-develop-brightgreen.svg)](https://pdimov.github.io/boostdep-report/develop/format.html) | [![Documentation](https://img.shields.io/badge/docs-develop-brightgreen.svg)](https://www.boost.org/doc/libs/develop/libs/format) | [![Enter the Matrix](https://img.shields.io/badge/matrix-develop-brightgreen.svg)](https://www.boost.org/development/tests/develop/developer/format.html)

### Directories

| Name        | Purpose                        |
| ----------- | ------------------------------ |
| `benchmark` | benchmark tool                 |
| `doc`       | documentation                  |
| `examples`  | use case examples              |
| `include`   | headers                        |
| `test`      | unit tests                     |
| `tools`     | development tools              |

### More information

* [Ask questions](https://stackoverflow.com/questions/ask?tags=c%2B%2B,boost,boost-format): Be sure to read the documentation first as Boost.Format, like any other string formatting library, has its own rules.
* [Report bugs](https://github.com/boostorg/format/issues): Be sure to mention Boost version, platform and compiler you're using. A small compilable code sample to reproduce the problem is always good as well.
* [Submit Pull Requests](https://github.com/boostorg/format/pulls) against the **develop** branch. Note that by submitting patches you agree to license your modifications under the [Boost Software License, Version 1.0](https://www.boost.org/LICENSE_1_0.txt).  Be sure to include tests proving your changes work properly.
* Discussions about the library are held on the [Boost developers mailing list](https://www.boost.org/community/groups.html#main). Be sure to read the [discussion policy](https://www.boost.org/community/policy.html) before posting and add the `[format]` tag at the beginning of the subject line.

