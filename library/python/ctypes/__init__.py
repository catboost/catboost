import ctypes


class StaticLibrary(ctypes.CDLL):
    """
    StaticLibrary('library name', {'symbol_name': symbol_addr, ...})  returns
    a ctypes.CDLL wrapper around the specified static symbols.  'library name'
    is only for display.  Symbol table is preserved in _syms for introspection.
    """
    def __init__(self, name, syms, **kwargs):
        ctypes.CDLL.__init__(self, name, handle=0, **kwargs)
        self._syms = syms

    def __getitem__(self, name_or_ordinal):
        ordinal = name_or_ordinal
        if not isinstance(ordinal, (int, long)):
            ordinal = self._syms[name_or_ordinal]
        func = self._FuncPtr(ordinal)
        if not isinstance(name_or_ordinal, (int, long)):
            func.__name__ = name_or_ordinal
        return func
