LIBRARY()



PEERDIR(
    library/cpp/colorizer
    library/overloaded
)

SRCS(
    completer.cpp
    completer_command.cpp
    completion_generator.cpp
    formatted_output.cpp
    last_getopt.cpp
    last_getopt_easy_setup.cpp
    last_getopt_opt.cpp
    last_getopt_opts.cpp
    last_getopt_parser.cpp
    last_getopt_parse_result.cpp
    modchooser.cpp
    opt.cpp
    opt2.cpp
    posix_getopt.cpp
    wrap.cpp
    ygetopt.cpp
)

END()
