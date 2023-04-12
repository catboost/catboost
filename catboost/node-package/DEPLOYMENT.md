# Package build & deployment procedure

## Building scripts

Scripts responsible for building and preparing the package for publishing are located in `/build_scripts` directory.

These scripts are built once on the first execution of any script.  To compile build scripts explicitly, run
```
npm run compile_build_scripts
```

## Scripts

- `npm run install` - main script for installing the package. Behaves differently if called from catboost directory or from the standalone package distribution:
  - If called from `catboost` repository, then builds the local version of `catboostmodel` library from source, compiles and links package against it. After that the package can be added locally via
  ```
    npm install $PATH_TO_CATBOOST_REPO/catboost/node-package
  ```
  - If called outside of the repository, then the package is considered to be installed from npm and it will be built and linked against the binaries specified in `config.json` file. This file is created during package preparation for publishing, see below.
- `npm run build` - build package locally in the `catboost` repository, link against library built from source.
- `npm run ci` - a single script for CI which runs the procedure described in "Release procedure" section.
- `npm run compile` - compile Typescript source files only.
- `npm run test` - run local unit tests.
- `npm run compile_build_scripts` - compile build scripts from Typescript sources.
- `npm run clean` - delete local artifacts.
- `npm run package_prepublish <version>` - prepare the package for publishing, does the following:
    1. Copies C headers required for building package on the client's side from repository to a `./inc` directory.
    2. Compiles Typescript source files.
    3. Fetches the binaries from the `github.com` releases for a given version (for ex. `v0.25.1`), prepopulates `config.json` file. This config file includes link to the binaries and sha256 checksums that will be verified on the client side. The binaries have to be downloaded during the package installation as they are too big to be part of the distributed package.

## Release procedure

0. Verify package locally (optional)
   1. Checkout the released branch.
   2. Run
        ```
        npm run build
        ```
        to build package from source.
   3. Run
        ```
        npm run test
        ```
        to verify that the tests are passing.
1. Run
    ```
    npm run package_prepublish <version>
    ```
2. Check generated `config.json` file, verify that the links and file checksums are correct.
3. Update package version in `package.json`.
4. Run integration deployment test (local docker required):
    ```
    npm run e2e
    ```
5. Publish package with
   ```
   npm publish
   ```

## CI

For setting up continuous integration the following had to be done:

1. Checkout the source code (at the release tag commit) on the host with Internet access and docker daemon running (with permissions to tag images and run containers).
2. From `catboost/catboost/node-package` subdirectory, execute ci script and check that it is executed correctly:
   ```
   npm run ci
   ```
3. Assuming that npm registry credentials are set up on the host, publish the package via
   ```
   npm publish
   ```
