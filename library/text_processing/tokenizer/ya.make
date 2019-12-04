

LIBRARY()

SRCS(
    options.cpp
    tokenizer.cpp
    GLOBAL lemmer_impl.cpp
)

PEERDIR(
    library/json
    library/langs
    library/object_factory
    library/tokenizer
)

IF (NOT CATBOOST_OPENSOURCE AND NOT NO_YANDEX_LEMMER)
    PEERDIR(
        library/text_processing/yandex_specific_lemmer
    )
ENDIF()

GENERATE_ENUM_SERIALIZATION(options.h)
GENERATE_ENUM_SERIALIZATION(lemmer_impl.h)

END()
