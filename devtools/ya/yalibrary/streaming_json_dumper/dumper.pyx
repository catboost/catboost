import io
import sys

cdef extern from "devtools/ya/yalibrary/streaming_json_dumper/lib/dump.h" namespace "NSJson":
    void DumpToStream(object obj, object stream) except *


def dump(obj, stream):
    DumpToStream(obj, stream)


def dumps(obj):
    stream = io.BytesIO()
    dump(obj, stream)
    return stream.getvalue()
