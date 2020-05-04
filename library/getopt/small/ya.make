LIBRARY()



PEERDIR(
    library/cpp/getopt/small
)

SRCS(
    completer_command.h
    completer.h
    completion_generator.h
    formatted_output.h
    last_getopt_easy_setup.h
    last_getopt.h
    last_getopt_handlers.h
    last_getopt_opt.h
    last_getopt_opts.h
    last_getopt_parse_result.h
    last_getopt_parser.h
    last_getopt_support.h
    modchooser.h
    opt2.h
    opt.h
    posix_getopt.h
    wrap.h
    ygetopt.h
)

END()
