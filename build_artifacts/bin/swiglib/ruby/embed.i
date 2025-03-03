%wrapper %{

#include <ruby.h>

int
main(argc, argv)
    int argc;
    char **argv;
{
    ruby_init();
    ruby_options(argc, argv);
    ruby_run();
    return 0;
}

%}
