## Test environments
* local Windows install, R 3.3.0
* local Ubuntu install, R 3.5.0
* rhub debian(devel), ubuntu(release), windows(devel)

## R CMD check results
There were no ERRORs or WARNINGs.

Two notes:
```
> checking installed package size ... NOTE
    installed size is 133.0Mb
    sub-directories of 1Mb or more:
      libs  132.7Mb
  NB: this package is only installed for sub-architecture 'x64'
```
Functionalities implemented in C++ with number of internal and 4rdparty libraries.
They all compiled into package library. Therefore it is large.

```
> checking compiled code ... NOTE
  Note: information on .o files for x64 is not available
```
Package requires customized build and Makefile is required instead of Makevars.

## Downstream dependencies
No issues detected

## Previous submission comments
  * Initial submission
