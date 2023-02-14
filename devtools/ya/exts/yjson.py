import json
import logging

loader = json
dumper = json

# Accelerate json.dump (dumping to a file object does not use fast c_encoder).
if True:

    def dump(obj, fp, **kwargs):
        fp.write(json.dumps(obj, **kwargs))

    dumper.dump = dump

# Use simplejson to prevent usage of a localy installed json library (/usr/lib/python/).
if True:
    try:
        import simplejson

        loader = simplejson
    except ImportError:
        pass

# Our library rocks: it is fast and can intern strings.
if True:
    try:
        import library.python.json as lpj

        _loads = lpj.loads

        def loads(s, **kwargs):
            kwargs['intern_keys'] = True
            kwargs['intern_vals'] = True
            kwargs['may_unicode'] = True
            return _loads(s, **kwargs)

        def load(fp, **kwargs):
            return loads(fp.read(), **kwargs)

        loader = lpj
        loader.loads = loads
        loader.load = load
    except ImportError:
        pass

# ultrajson library is fast in dumping and supports sort_keys and indentation.
if True:
    try:
        import ujson

        _dump = ujson.dump
        _dumps = ujson.dumps

        _fallback_dump = dumper.dump
        _fallback_dumps = dumper.dumps

        def dump(obj, fp, **kwargs):
            if 'separators' in kwargs or 'cls' in kwargs:
                return _fallback_dump(obj, fp, **kwargs)
            else:
                kwargs.pop('default', None)
                kwargs.pop('encoding', None)
                kwargs['escape_forward_slashes'] = False
                kwargs['double_precision'] = 15
                try:
                    return _dump(obj, fp, **kwargs)
                except OverflowError:
                    fallback_kwargs = kwargs.copy()
                    del fallback_kwargs['escape_forward_slashes']
                    del fallback_kwargs['double_precision']
                    logging.exception("While dumping object %s %s", id(obj), type(obj))
                    logging.info("Using fallback dumper")
                    return _fallback_dump(obj, fp, **fallback_kwargs)

        def dumps(obj, **kwargs):
            if 'separators' in kwargs or 'cls' in kwargs:
                return _fallback_dumps(obj, **kwargs)
            else:
                kwargs.pop('default', None)
                kwargs.pop('encoding', None)
                kwargs['escape_forward_slashes'] = False
                kwargs['double_precision'] = 15
                try:
                    return _dumps(obj, **kwargs)
                except OverflowError:
                    fallback_kwargs = kwargs.copy()
                    del fallback_kwargs['escape_forward_slashes']
                    del fallback_kwargs['double_precision']
                    logging.exception("While dumping object %s %s", id(obj), type(obj))
                    logging.info("Using fallback dumper")
                    return _fallback_dumps(obj, **fallback_kwargs)

        dumper = ujson
        dumper.dump = dump
        dumper.dumps = dumps
    except ImportError:
        pass


loads = loader.loads
load = loader.load

dumps = dumper.dumps
dump = dumper.dump

JSONEncoder = json.JSONEncoder
