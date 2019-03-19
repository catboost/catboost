## Test environments
* local Windows install, R 3.5.1 (2018-07-02)
* local Ubuntu install, R 3.2.3 (2015-12-10)
* local Darwin install, R 3.5.2 (2018-12-20)

## Results of `R CMD check catboost_0.13.tar.gz`

### Windows
0 ERRORS, 0 WARNINGS, 2 NOTES:
```
* checking installed package size ... NOTE
  installed size is 139.7Mb
  sub-directories of 1Mb or more:
    libs  139.3Mb
* checking compiled code ... NOTE
Note: information on .o files for x64 is not available
```
Functionalities implemented in C++ with number of internal and 3rd party libraries.
They all compiled into package library. Therefore it is large.

Package requires customized build and Makefile is required instead of Makevars.

### Ubuntu (with `--no-manual`)
0 ERRORS, 0 WARNINGS, 2 NOTES:
```
* checking installed package size ... NOTE
  installed size is 141.6Mb
  sub-directories of 1Mb or more:
    libs  141.2Mb
* checking compiled code ... NOTE
Note: information on .o files is not available
File ‘/place/home/dbakshee/catboost/catboost/R-package/catboost.Rcheck/catboost/libs/libcatboostr.so’:
  Found ‘abort’, possibly from ‘abort’ (C)
  Found ‘exit’, possibly from ‘exit’ (C)
  Found ‘printf’, possibly from ‘printf’ (C)
  Found ‘puts’, possibly from ‘printf’ (C), ‘puts’ (C)
  Found ‘stderr’, possibly from ‘stderr’ (C)
  Found ‘stdout’, possibly from ‘stdout’ (C)
```

### Darwin (with `LC_ALL=en_US.UTF-8` and `--no-manual`)
0 ERRORS, 0 WARNINGS, 2 NOTES:
```
* checking installed package size ... NOTE
  installed size is 26.7Mb
  sub-directories of 1Mb or more:
    libs  26.3Mb
* checking compiled code ... NOTE
Note: information on .o files is not available
File ‘/Users/zomb-ml-platform-msk/catboost/catboost/R-package/catboost.Rcheck/catboost/libs/libcatboostr.so’:
  Found ‘___assert_rtn’, possibly from ‘assert’ (C)
  Found ‘___stderrp’, possibly from ‘stderr’ (C)
  Found ‘___stdoutp’, possibly from ‘stdout’ (C)
  Found ‘_abort’, possibly from ‘abort’ (C)
  Found ‘_exit’, possibly from ‘exit’ (C)
  Found ‘_printf’, possibly from ‘printf’ (C)
  Found ‘_puts’, possibly from ‘printf’ (C), ‘puts’ (C)
```

## Downstream dependencies
* No issues detected

## Previous submission comments
* Initial submission
