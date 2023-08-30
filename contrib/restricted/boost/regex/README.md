Boost Regex Library
============================

The Boost Regex library provides regular expression support for C++, this library is the ancestor to std::regex and still goes beyond
and offers some advantages to, the standard version.

The full documentation is available on [boost.org](http://www.boost.org/doc/libs/release/libs/regex/index.html).

## Standalone Mode ##

 This library may now be used in "standalone" mode without the rest of the Boost C++ libraries, in order to do this you must either:

* Have a C++17 compiler that supports __has_include, in this case if <boost/config.hpp> is not present then the library will automoatically enter standalone mode. Or:
* Define BOOST_REGEX_STANDALONE when building.

The main difference between the 2 modes, is that when Boost.Config is present the library will automatically configure itself around various compiler defects. In particular in order to use the library with exception support turned off, you will either need a copy of Boost.Config in your include path, or else manually define BOOST_NO_EXCEPTIONS when building.

In any event, to obtain a standalone version of this library, simply download a .zip of the "master" branch of this repository.

## Support, bugs and feature requests ##

Bugs and feature requests can be reported through the [Gitub issue tracker](https://github.com/boostorg/regex/issues)
(see [open issues](https://github.com/boostorg/regex/issues) and
[closed issues](https://github.com/boostorg/regex/issues?utf8=%E2%9C%93&q=is%3Aissue+is%3Aclosed)).

You can submit your changes through a [pull request](https://github.com/boostorg/regex/pulls).

There is no mailing-list specific to Boost Regex, although you can use the general-purpose Boost [mailing-list](http://lists.boost.org/mailman/listinfo.cgi/boost-users) using the tag [regex].


## Development ##

Clone the whole boost project, which includes the individual Boost projects as submodules ([see boost+git doc](https://github.com/boostorg/boost/wiki/Getting-Started)): 

    git clone https://github.com/boostorg/boost
    cd boost
    git submodule update --init

The Boost Regex Library is located in `libs/regex/`. 

### Running tests ###
First, make sure you are in `libs/regex/test`. 
You can either run all the tests listed in `Jamfile.v2` or run a single test:

    ../../../b2                        <- run all tests
    ../../../b2 regex_regress          <- single test

