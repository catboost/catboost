from toml import TomlDecoder as TomlDecoder, TomlEncoder as TomlEncoder

class TomlOrderedDecoder(TomlDecoder):
    def __init__(self) -> None: ...

class TomlOrderedEncoder(TomlEncoder):
    def __init__(self) -> None: ...
