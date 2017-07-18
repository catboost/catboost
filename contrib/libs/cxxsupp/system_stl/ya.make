LIBRARY()



NO_PLATFORM()

ADDINCL(GLOBAL contrib/libs/cxxsupp/system_stl/include)

LDFLAGS(
    -lgcc_s
    -lstdc++
)

END()
