UNITTEST()



SRCS(
    alignment_ut.cpp
)

ARCHIVE_ASM(
    NAME ArchiveAsm DONTCOMPRESS
    data_file.txt
    data_file2.txt
)

ARCHIVE(
    NAME simple_archive.inc DONTCOMPRESS
    data_file.txt
    data_file2.txt
)


PEERDIR(
    library/archive
)


END()
