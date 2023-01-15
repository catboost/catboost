

LIBRARY()

SRCS(
    options.cpp
    tokenizer.cpp
    GLOBAL lemmer_impl.cpp
)

PEERDIR(
    library/cpp/json
    library/langs
    library/object_factory
    library/cpp/tokenizer
)

IF (NOT CATBOOST_OPENSOURCE AND NOT NO_YANDEX_LEMMER)
    PEERDIR(
        library/cpp/text_processing/yandex_specific_lemmer
    )
ENDIF()

GENERATE_ENUM_SERIALIZATION(options.h)
GENERATE_ENUM_SERIALIZATION(lemmer_impl.h)

END()
