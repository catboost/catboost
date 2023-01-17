JSON writer with no external dependencies, producing output
where HTML special characters are always escaped.

Use it like this:

    #include <library/cpp/json/writer/json.h>
    ...

    NJsonWriter::TBuf json;
    json.BeginList()
        .WriteString("<script>")
        .EndList();
    Cout << json.Str(); // output: ["\u003Cscript\u003E"]

For compatibility with legacy formats where object keys
are not quoted, use CompatWriteKeyWithoutQuotes:
    
    NJsonWriter::TBuf json;
    json.BeginObject()
        .CompatWriteKeyWithoutQuotes("r").WriteInt(1)
        .CompatWriteKeyWithoutQuotes("n").WriteInt(0)
    .EndObject();
    Cout << json.Str(); // output: {r:1,n:0}
