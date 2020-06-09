UNITTEST()



SRCS(
    test_port_manager.cpp
)

SIZE(LARGE)

# We need to run tests at the same time on the single machine
FORK_SUBTESTS()

TAG(
    ya:fat
    ya:force_sandbox
)

END()
