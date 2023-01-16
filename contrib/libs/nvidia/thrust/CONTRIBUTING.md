# Table of Contents

1. [Contributing to Thrust](#contributing-to-thrust)
1. [CMake Options](#cmake-options)
1. [Development Model](#development-model)

# Contributing to Thrust

Thrust uses Github to manage all open-source development, including bug
tracking, pull requests, and design discussions. This document details how to get
started as a Thrust contributor.

An overview of this process is:

1. [Clone the Thrust repository](#clone-the-thrust-repository)
1. [Setup a fork of Thrust](#setup-a-fork-of-thrust)
1. [Setup your environment](#setup-your-environment)
1. [Create a development branch](#create-a-development-branch)
1. [Local development loop](#local-development-loop)
1. [Push development branch to your fork](#push-development-branch-to-your-fork)
1. [Create pull request](#create-pull-request)
1. [Address feedback and update pull request](#address-feedback-and-update-pull-request)
1. [When your PR is approved...](#when-your-pr-is-approved)

## Clone the Thrust Repository

To get started, clone the main repository to your local computer. Thrust should
be cloned recursively to setup the CUB submodule (required for `CUDA`
acceleration).

```
git clone --recursive https://github.com/NVIDIA/thrust.git
cd thrust
```

## Setup a Fork of Thrust

You'll need a fork of Thrust on Github to create a pull request. To setup your
fork:

1. Create a Github account (if needed)
2. Go to [the Thrust Github page](https://github.com/NVIDIA/thrust)
3. Click "Fork" and follow any prompts that appear.

Once your fork is created, setup a new remote repo in your local Thrust clone:

```
git remote add github-fork git@github.com:<GITHUB_USERNAME>/thrust.git
```

If you need to modify CUB, too, go to
[the CUB Github page](https://github.com/NVIDIA/cub) and repeat this process.
Create CUB's `github-fork` remote in the `thrust/dependencies/cub` submodule.

## Setup Your Environment

### Git Environment

If you haven't already, this is a good time to tell git who you are. This
information is used to fill out authorship information on your git commits.

```
git config --global user.name "John Doe"
git config --global user.email johndoe@example.com
```

### Configure CMake builds

Thrust uses [CMake](https://www.cmake.org) for its primary build system. To
configure, build, and test your checkout of Thrust:

```
# Create build directory:
mkdir build
cd build

# Configure -- use one of the following:
cmake ..                                 # Command line interface
cmake -DTHRUST_INCLUDE_CUB_CMAKE=ON ..   # Enables CUB development targets
ccmake ..                # ncurses GUI (Linux only)
cmake-gui                # Graphical UI, set source/build directories in the app

# Build:
cmake --build . -j <num jobs>   # invokes make (or ninja, etc)

# Run tests and examples:
ctest
```

See [CMake Options](#cmake-options) for details on customizing the build. To
enable CUB tests and examples, set the `THRUST_INCLUDE_CUB_CMAKE` option to
`ON`. Additional CMake options for CUB are listed
[here](https://github.com/NVIDIA/cub/blob/main/CONTRIBUTING.md#cmake-options).

## Create a Development Branch

All work should be done in a development branch (also called a "topic branch")
and not directly in the `main` branch. This makes it easier to manage multiple
in-progress patches at once, and provides a descriptive label for your patch
as it passes through the review system.

To create a new branch based on the current `main`:

```
# Checkout local main branch:
cd /path/to/thrust/sources
git checkout main

# Sync local main branch with github:
git pull

# Create a new branch named `my_descriptive_branch_name` based on main:
git checkout -b my_descriptive_branch_name

# Verify that the branch has been created and is currently checked out:
git branch
```

Thrust branch names should follow a particular pattern:

- For new features, name the branch `feature/<name>`
- For bugfixes associated with a github issue, use `bug/github/<bug-description>-<bug-id>`
  - Internal nvidia and gitlab bugs should use `nvidia` or `gitlab` in place of
    `github`.

If you plan to work on CUB as part of your patch, repeat this process in the
`thrust/dependencies/cub` submodule.

## Local Development Loop

### Edit, Build, Test, Repeat

Once the topic branch is created, you're all set to start working on Thrust
code. Make some changes, then build and test them:

```
# Implement changes:
cd /path/to/thrust/sources
emacs thrust/some_file.h # or whatever editor you prefer

# Create / update a unit test for your changes:
emacs testing/some_test.cu

# Check that everything builds and tests pass:
cd /path/to/thrust/build/directory
cmake --build . -j <num jobs>
ctest
```

### Creating a Commit

Once you're satisfied with your patch, commit your changes:

#### Thrust-only Changes

```
# Manually add changed files and create a commit:
cd /path/to/thrust
git add thrust/some_file.h
git add testing/some_test.cu
git commit

# Or, if possible, use git-gui to review your changes while building your patch:
git gui
```

#### Thrust and CUB Changes

```
# Create CUB patch first:
cd /path/to/thrust/dependencies/cub
# Manually add changed files and create a commit:
git add cub/some_file.cuh
git commit

# Create Thrust patch, including submodule update:
cd /path/to/thrust/
git add dependencies/cub # Updates submodule info
git add thrust/some_file.h
git add testing/some_test.cu
git commit

# Or, if possible, use git-gui to review your changes while building your patch:
cd /path/to/thrust/dependencies/cub
git gui
cd /path/to/thrust
git gui # Include dependencies/cub as part of your commit

```

#### Writing a Commit Message

Your commit message will communicate the purpose and rationale behind your
patch to other developers, and will be used to populate the initial description
of your Github pull request.

When writing a commit message, the following standard format should be used,
since tools in the git ecosystem are designed to parse this correctly:

```
First line of commit message is a short summary (<80 char)
<Second line left blank>
Detailed description of change begins on third line. This portion can
span multiple lines, try to manually wrap them at something reasonable.

Blank lines can be used to separate multiple paragraphs in the description.

If your patch is associated with another pull request or issue in the main
Thrust repository, you should reference it with a `#` symbol, e.g.
#1023 for issue 1023.

For issues / pull requests in a different github repo, reference them using
the full syntax, e.g. NVIDIA/cub#4 for issue 4 in the NVIDIA/cub repo.

Markdown is recommended for formatting more detailed messages, as these will
be nicely rendered on Github, etc.
```

## Push Development Branch to your Fork

Once you've committed your changes to a local development branch, it's time to
push them to your fork:

```
cd /path/to/thrust/checkout
git checkout my_descriptive_branch_name # if not already checked out
git push --set-upstream github-fork my_descriptive_branch_name
```

`--set-upstream github-fork` tells git that future pushes/pulls on this branch
should target your `github-fork` remote by default.

If have CUB changes to commit as part of your patch, repeat this process in the
`thrust/dependencies/cub` submodule.

## Create Pull Request

To create a pull request for your freshly pushed branch, open your github fork
in a browser by going to `https://www.github.com/<GITHUB_USERNAME>/thrust`. A
prompt may automatically appear asking you to create a pull request if you've
recently pushed a branch.

If there's no prompt, go to "Code" > "Branches" and click the appropriate
"New pull request" button for your branch.

If you would like a specific developer to review your patch, feel free to
request them as a reviewer at this time.

The Thrust team will review your patch, test it on NVIDIA's internal CI, and
provide feedback.


If have CUB changes to commit as part of your patch, repeat this process with
your CUB branch and fork.

## Address Feedback and Update Pull Request

If the reviewers request changes to your patch, use the following process to
update the pull request:

```
# Make changes:
cd /path/to/thrust/sources
git checkout my_descriptive_branch_name
emacs thrust/some_file.h
emacs testing/some_test.cu

# Build + test
cd /path/to/thrust/build/directory
cmake --build . -j <num jobs>
ctest

# Amend commit:
cd /path/to/thrust/sources
git add thrust/some_file.h
git add testing/some_test.cu
git commit --amend
# Or
git gui # Check the "Amend Last Commit" box

# Update the branch on your fork:
git push -f
```

At this point, the pull request should show your recent changes.

If have CUB changes to commit as part of your patch, repeat this process in the
`thrust/dependencies/cub` submodule, and be sure to include any CUB submodule
updates as part of your commit.

## When Your PR is Approved

Once your pull request is approved by the Thrust team, no further action is
needed from you. We will handle integrating it since we must coordinate changes
to `main` with NVIDIA's internal perforce repository.

# CMake Options

A Thrust build is configured using CMake options. These may be passed to CMake
using

```
cmake -D<option_name>=<value> /path/to/thrust/sources
```

or configured interactively with the `ccmake` or `cmake-gui` interfaces.

Thrust supports two build modes. By default, a single configuration is built
that targets a specific host system, device system, and C++ dialect.
When `THRUST_ENABLE_MULTICONFIG` is `ON`, multiple configurations
targeting a variety of systems and dialects are generated.

The CMake options are divided into these categories:

1. [Generic CMake Options](#generic-cmake-options): Options applicable to all
   Thrust builds.
1. [Single Config CMake Options](#single-config-cmake-options) Options
   applicable only when `THRUST_ENABLE_MULTICONFIG` is disabled.
1. [Multi Config CMake Options](#multi-config-cmake-options) Options applicable
   only when `THRUST_ENABLE_MULTICONFIG` is enabled.
1. [CUDA Specific CMake Options](#cuda-specific-cmake-options) Options that
   control CUDA compilation. Only available when one or more configurations
   targets the CUDA system.
1. [TBB Specific CMake Options](#tbb-specific-cmake-options) Options that
   control TBB compilation. Only available when one or more configurations
   targets the TBB system.

## Generic CMake Options

- `CMAKE_BUILD_TYPE={Release, Debug, RelWithDebInfo, MinSizeRel}`
  - Standard CMake build option. Default: `RelWithDebInfo`
- `THRUST_ENABLE_HEADER_TESTING={ON, OFF}`
  - Whether to test compile public headers. Default is `ON`.
- `THRUST_ENABLE_TESTING={ON, OFF}`
  - Whether to build unit tests. Default is `ON`.
- `THRUST_ENABLE_EXAMPLES={ON, OFF}`
  - Whether to build examples. Default is `ON`.
- `THRUST_ENABLE_MULTICONFIG={ON, OFF}`
  - Toggles single-config and multi-config modes. Default is `OFF` (single config).
- `THRUST_ENABLE_EXAMPLE_FILECHECK={ON, OFF}`
  - Enable validation of example outputs using the LLVM FileCheck utility.
    Default is `OFF`.
- `THRUST_ENABLE_INSTALL_RULES={ON, OFF}`
  - If true, installation rules will be generated for thrust. Default is `ON`.

## Single Config CMake Options

- `THRUST_HOST_SYSTEM={CPP, TBB, OMP}`
  - Selects the host system. Default: `CPP`
- `THRUST_DEVICE_SYSTEM={CUDA, TBB, OMP, CPP}`
  - Selects the device system. Default: `CUDA`
- `THRUST_CPP_DIALECT={11, 14, 17}`
  - Selects the C++ standard dialect to use. Default is `14` (C++14).

## Multi Config CMake Options

- `THRUST_MULTICONFIG_ENABLE_DIALECT_CPPXX={ON, OFF}`
  - Toggle whether a specific C++ dialect will be targeted.
  - Possible values of `XX` are `{11, 14, 17}`.
  - By default, only C++14 is enabled.
- `THRUST_MULTICONFIG_ENABLE_SYSTEM_XXXX={ON, OFF}`
  - Toggle whether a specific system will be targeted.
  - Possible values of `XXXX` are `{CPP, CUDA, TBB, OMP}`
  - By default, only `CPP` and `CUDA` are enabled.
- `THRUST_MULTICONFIG_WORKLOAD={SMALL, MEDIUM, LARGE, FULL}`
  - Restricts the host/device combinations that will be targeted.
  - By default, the `SMALL` workload is used.
  - The full cross product of `host x device` systems results in 12
    configurations, some of which are more important than others.
    This option can be used to prune some of the less important ones.
  - `SMALL`: (3 configs) Minimal coverage and validation of each device system against the `CPP` host.
  - `MEDIUM`: (6 configs) Cheap extended coverage.
  - `LARGE`: (8 configs) Expensive extended coverage. Includes all useful build configurations.
  - `FULL`: (12 configs) The complete cross product of all possible build configurations.

| Config   | Workloads | Value      | Expense   | Note                         |
|----------|-----------|------------|-----------|------------------------------|
| CPP/CUDA | `F L M S` | Essential  | Expensive | Validates CUDA against CPP   |
| CPP/OMP  | `F L M S` | Essential  | Cheap     | Validates OMP against CPP    |
| CPP/TBB  | `F L M S` | Essential  | Cheap     | Validates TBB against CPP    |
| CPP/CPP  | `F L M  ` | Important  | Cheap     | Tests CPP as device          |
| OMP/OMP  | `F L M  ` | Important  | Cheap     | Tests OMP as host            |
| TBB/TBB  | `F L M  ` | Important  | Cheap     | Tests TBB as host            |
| TBB/CUDA | `F L    ` | Important  | Expensive | Validates TBB/CUDA interop   |
| OMP/CUDA | `F L    ` | Important  | Expensive | Validates OMP/CUDA interop   |
| TBB/OMP  | `F      ` | Not useful | Cheap     | Mixes CPU-parallel systems   |
| OMP/TBB  | `F      ` | Not useful | Cheap     | Mixes CPU-parallel systems   |
| TBB/CPP  | `F      ` | Not Useful | Cheap     | Parallel host, serial device |
| OMP/CPP  | `F      ` | Not Useful | Cheap     | Parallel host, serial device |

## CUDA Specific CMake Options

- `THRUST_INCLUDE_CUB_CMAKE={ON, OFF}`
  - If enabled, the CUB project will be built as part of Thrust. Default is
    `OFF`.
  - This adds CUB tests, etc. Useful for working on both CUB and Thrust
    simultaneously.
  - CUB configurations will be generated for each C++ dialect targeted by
    the current Thrust build.
- `THRUST_INSTALL_CUB_HEADERS={ON, OFF}`
  - If enabled, the CUB project's headers will be installed through Thrust's
    installation rules. Default is `ON`.
  - This option depends on `THRUST_ENABLE_INSTALL_RULES`.
- `THRUST_ENABLE_COMPUTE_XX={ON, OFF}`
  - Controls the targeted CUDA architecture(s)
  - Multiple options may be selected when using NVCC as the CUDA compiler.
  - Valid values of `XX` are:
    `{35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80}`
  - Default value depends on `THRUST_DISABLE_ARCH_BY_DEFAULT`:
- `THRUST_ENABLE_COMPUTE_FUTURE={ON, OFF}`
  - If enabled, CUDA objects will target the most recent virtual architecture
    in addition to the real architectures specified by the
    `THRUST_ENABLE_COMPUTE_XX` options.
  - Default value depends on `THRUST_DISABLE_ARCH_BY_DEFAULT`:
- `THRUST_DISABLE_ARCH_BY_DEFAULT={ON, OFF}`
  - When `ON`, all `THRUST_ENABLE_COMPUTE_*` options are initially `OFF`.
  - Default: `OFF` (meaning all architectures are enabled by default)
- `THRUST_ENABLE_TESTS_WITH_RDC={ON, OFF}`
  - Whether to enable Relocatable Device Code when building tests.
    Default is `OFF`.
- `THRUST_ENABLE_EXAMPLES_WITH_RDC={ON, OFF}`
  - Whether to enable Relocatable Device Code when building examples.
    Default is `OFF`.

## TBB Specific CMake Options

- `THRUST_TBB_ROOT=<path to tbb root>`
  - When the TBB system is requested, set this to the root of the TBB installation
    (e.g. the location of `lib/`, `bin/` and `include/` for the TBB libraries).

# Development Model

The following is a description of the basic development process that Thrust follows. This is a living
document that will evolve as our process evolves.

Thrust is distributed in three ways:

   * On GitHub.
   * In the NVIDIA HPC SDK.
   * In the CUDA Toolkit.

## Trunk Based Development

Thrust uses [trunk based development](https://trunkbaseddevelopment.com). There is a single long-lived
branch called `main`. Engineers may create branches for feature development. Such branches always
merge into `main`. There are no release branches. Releases are produced by taking a snapshot of
`main` ("snapping"). After a release has been snapped from `main`, it will never be changed.

## Repositories

As Thrust is developed both on GitHub and internally at NVIDIA, there are three main places where code lives:

   * The Source of Truth, the [public Thrust repository](https://github.com/NVIDIA/thrust), referred to as
     `github` later in this document.
   * An internal GitLab repository, referred to as `gitlab` later in this document.
   * An internal Perforce repository, referred to as `perforce` later in this document.

## Versioning

Thrust has its own versioning system for releases, independent of the versioning scheme of the NVIDIA
HPC SDK or the CUDA Toolkit.

Today, Thrust version numbers have a specific [semantic meaning](https://semver.org/).
Releases prior to 1.10.0 largely, but not strictly, followed these semantic meanings.

The version number for a Thrust release uses the following format: `MMM.mmm.ss-ppp`, where:

   * `THRUST_VERSION_MAJOR`/`MMM`: Major version, up to 3 decimal digits. It is incremented
     when changes that are API-backwards-incompatible are made.
   * `THRUST_VERSION_MINOR`/`mmm`: Minor version, up to 3 decimal digits. It is incremented when
     breaking API, ABI, or semantic changes are made.
   * `THRUST_VERSION_SUBMINOR`/`ss`: Subminor version, up to 2 decimal digits. It is incremented
     when notable new features or bug fixes or features that are API-backwards-compatible are made.
   * `THRUST_PATCH_NUMBER`/`ppp`: Patch number, up to 3 decimal digits. This is no longer used and
     will be zero for all future releases.

The `<thrust/version.h>` header defines `THRUST_*` macros for all of the version components mentioned
above. Additionally, a `THRUST_VERSION` macro is defined, which is an integer literal containing all
of the version components except for `THRUST_PATCH_NUMBER`.

## Branches and Tags

The following tag names are used in the Thrust project:

  * `github/nvhpc-X.Y`: the tag that directly corresponds to what has been shipped in the NVIDIA HPC SDK release X.Y.
  * `github/cuda-X.Y`: the tag that directly corresponds to what has been shipped in the CUDA Toolkit release X.Y.
  * `github/A.B.C`: the tag that directly corresponds to Thrust version A.B.C.
  * `github/A.B.C-rcN`: the tag that directly corresponds to Thrust version A.B.C release candidate N.

The following branch names are used in the Thrust project:

  * `github/main`: the Source of Truth development branch of Thrust.
  * `github/old-master`: the old Source of Truth branch, before unification of public and internal repositories.
  * `github/feature/<name>`: feature branch for a feature under development.
  * `github/bug/<bug-system>/<bug-description>-<bug-id>`: bug fix branch, where `bug-system` is `github` or `nvidia`.
  * `gitlab/main`: mirror of `github/main`.
  * `perforce/private`: mirrored `github/main`, plus files necessary for internal NVIDIA testing systems.

On the rare occasion that we cannot do work in the open, for example when developing a change specific to an
unreleased product, these branches may exist on `gitlab` instead of `github`. By default, everything should be
in the open on `github` unless there is a strong motivation for it to not be open.

# Release Process

This section is a work in progress.

## Update Compiler Explorer

Thrust and CUB are bundled together on
[Compiler Explorer](https://www.godbolt.org/) (CE) as libraries for the CUDA
language. When releasing a new version of these projects, CE will need to be
updated.

There are two files in two repos that need to be updated:

### libraries.yaml

- Repo: https://github.com/compiler-explorer/infra
- Path: bin/yaml/libraries.yaml

This file tells CE how to pull in library files and defines which versions to
fetch. Look for the `thrustcub:` section:

```yaml
    thrustcub:
      type: github
      method: clone_branch
      repo: NVIDIA/thrust
      check_file: dependencies/cub/cub/cub.cuh
      targets:
        - 1.9.9
        - 1.9.10
        - 1.9.10-1
        - 1.10.0
```

Simply add the new version tag to list of `targets:`. This will check out the
specified tag to `/opt/compiler-explorer/libs/thrustcub/<tag>/`.

### cuda.amazon.properties

- Repo: https://github.com/compiler-explorer/compiler-explorer
- File: etc/config/cuda.amazon.properties

This file defines the library versions displayed in the CE UI and maps them
to a set of include directories. Look for the `libs.thrustcub` section:

```yaml
libs.thrustcub.name=Thrust+CUB
libs.thrustcub.description=CUDA collective and parallel algorithms
libs.thrustcub.versions=trunk:109090:109100:109101:110000
libs.thrustcub.url=http://www.github.com/NVIDIA/thrust
libs.thrustcub.versions.109090.version=1.9.9
libs.thrustcub.versions.109090.path=/opt/compiler-explorer/libs/thrustcub/1.9.9:/opt/compiler-explorer/libs/thrustcub/1.9.9/dependencies/cub
libs.thrustcub.versions.109100.version=1.9.10
libs.thrustcub.versions.109100.path=/opt/compiler-explorer/libs/thrustcub/1.9.10:/opt/compiler-explorer/libs/thrustcub/1.9.10/dependencies/cub
libs.thrustcub.versions.109101.version=1.9.10-1
libs.thrustcub.versions.109101.path=/opt/compiler-explorer/libs/thrustcub/1.9.10-1:/opt/compiler-explorer/libs/thrustcub/1.9.10-1/dependencies/cub
libs.thrustcub.versions.110000.version=1.10.0
libs.thrustcub.versions.110000.path=/opt/compiler-explorer/libs/thrustcub/1.10.0:/opt/compiler-explorer/libs/thrustcub/1.10.0/dependencies/cub
libs.thrustcub.versions.trunk.version=trunk
libs.thrustcub.versions.trunk.path=/opt/compiler-explorer/libs/thrustcub/trunk:/opt/compiler-explorer/libs/thrustcub/trunk/dependencies/cub
```

Add a new version identifier to the `libs.thrustcub.versions` key, using the
convention `X.Y.Z-W -> XXYYZZWW`. Then add a corresponding UI label (the
`version` key) and set of colon-separated include paths for Thrust and CUB
(`path`). The version used in the `path` entries must exactly match the tag
specified in `libraries.yaml`.
