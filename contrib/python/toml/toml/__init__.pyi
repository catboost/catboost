from toml import decoder as decoder, encoder as encoder

load = decoder.load
loads = decoder.loads
TomlDecoder = decoder.TomlDecoder
TomlDecodeError = decoder.TomlDecodeError
TomlPreserveCommentDecoder = decoder.TomlPreserveCommentDecoder
dump = encoder.dump
dumps = encoder.dumps
TomlEncoder = encoder.TomlEncoder
TomlArraySeparatorEncoder = encoder.TomlArraySeparatorEncoder
TomlPreserveInlineDictEncoder = encoder.TomlPreserveInlineDictEncoder
TomlNumpyEncoder = encoder.TomlNumpyEncoder
TomlPreserveCommentEncoder = encoder.TomlPreserveCommentEncoder
TomlPathlibEncoder = encoder.TomlPathlibEncoder
