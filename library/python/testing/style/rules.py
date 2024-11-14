import os
import six


def style_required(path, data, skip_links=True):
    if get_skip_reason(path, data, skip_links):
        return False
    return True


def get_skip_reason(path, data, skip_links=True):
    return _path_skip_reason(path, skip_links) or _content_skip_reason(path, data)


def _path_skip_reason(path, skip_links=True):
    if '/generated/' in path:
        return "path '{}' contains '/generated/'".format(path)

    if path and '/contrib/' in path and '/.yandex_meta/' not in path:
        return "path '{}' contains '/contrib/'".format(path)

    if path and '/vendor/' in path:
        return "path '{}' contains '/vendor/'".format(path)

    if skip_links and os.path.islink(path):
        return "path '{}' is a symlink".format(path)


def _content_skip_reason(path, data):
    if not isinstance(data, six.string_types):
        data = data.decode()

    for substr in [
        '# DO_NOT_STYLE',
        '// DO_NOT_STYLE',
        'THIS SOFTWARE',
        'WITHOUT WARRANT',  # WARRANTY, WARRANTIES
    ]:
        if substr in data:
            return "file '{}' contains '{}'".format(path, substr)
