LIBRARY()



SRCS(
    flat_hash.cpp
    lib/concepts/container.cpp
    lib/concepts/iterator.cpp
    lib/concepts/size_fitter.cpp
    lib/concepts/value_marker.cpp
	lib/containers.cpp
    lib/expanders.cpp
    lib/iterator.cpp
    lib/map.cpp
    lib/probings.cpp
    lib/set.cpp
    lib/size_fitters.cpp
    lib/table.cpp
    lib/value_markers.cpp
)

END()

RECURSE(
    benchmark
    fuzz
    ut
)
