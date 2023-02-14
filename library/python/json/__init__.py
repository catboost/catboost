from library.python.json.loads import loads as _loads
from simplejson import loads as _sj_loads


def loads(*args, **kwargs):
    try:
        return _loads(*args, **kwargs)
    except Exception as e:
        if 'invalid syntax at token' in str(e):
            kwargs.pop('intern_keys', None)
            kwargs.pop('intern_vals', None)
            kwargs.pop('may_unicode', None)
            return _sj_loads(*args, **kwargs)

        raise


from simplejson import load, dump, dumps  # noqa


def read_file(file_name, **kwargs):
    """
    Read file and return its parsed json contents.

    All kwargs will be proxied to `json.load` method as is.

    :param file_name: file with json contents
    :return: parsed json contents
    """
    with open(file_name) as f:
        return load(f, **kwargs)


def write_file(file_name, contents, **kwargs):
    """
    Dump json data to file.

    All kwargs will be proxied to `json.dump` method as is.

    :param file_name: file to dump to
    :param contents: JSON-serializable object
    """
    with open(file_name, "w") as f:
        dump(contents, f, **kwargs)
