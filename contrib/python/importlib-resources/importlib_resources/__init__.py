import io
import pkgutil
from contextlib import contextmanager

__all__ = 'read_binary read_text open_binary open_text is_resource contents path'.split()

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = OSError


def read_binary(package, resource):
    data = pkgutil.get_data(package, resource)
    if data is None:
        raise FileNotFoundError('{} does not contain {!r}'.format(package, resource))
    return data


def read_text(package, resource, encoding='utf-8', errors='strict'):
    return read_binary(package, resource).decode(encoding, errors)


def open_binary(package, resource):
    return io.BytesIO(read_binary(package, resource))


def open_text(package, resource, encoding='utf-8', errors='strict'):
    return io.StringIO(read_text(package, resource, encoding, errors))


def is_resource(package, name):
    try:
        read_binary(package, name)
        return True
    except (FileNotFoundError, OSError, IOError):
        return False


def contents(package):
    raise NotImplementedError('importlib_resources.contents is not implemented')


@contextmanager
def path(package, resource):
    raise NotImplementedError('importlib_resources.path is not implemented')
    yield None
