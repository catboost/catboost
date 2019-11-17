import sys
import hashlib
import base64


class Result(object):
    pass


def lazy(func):
    result = Result()

    def wrapper():
        try:
            return result._result
        except AttributeError:
            result._result = func()

        return result._result

    return wrapper


def pathid(path):
    return base64.b32encode(hashlib.md5(path).digest()).lower().strip('=')


def listid(l):
    return pathid(str(sorted(l)))


def unpair(lst):
    for x, y in lst:
        yield x
        yield y


def iterpair(lst):
    y = None

    for x in lst:
        if y:
            yield (y, x)

            y = None
        else:
            y = x


def stripext(fname):
    return fname[:fname.rfind('.')]


def tobuilddir(fname):
    if not fname:
        return '$B'
    if fname.startswith('$S'):
        return fname.replace('$S', '$B', 1)
    else:
        return fname


def before(s, ss):
    p = s.find(ss)

    if p == -1:
        return s

    return s[:p]


def sort_by_keywords(keywords, args):
    flat = []
    res = {}

    cur_key = None
    limit = -1
    for arg in args:
        if arg in keywords:
            limit = keywords[arg]
            if limit == 0:
                res[arg] = True
                cur_key = None
                limit = -1
            else:
                cur_key = arg
            continue
        if limit == 0:
            cur_key = None
            limit = -1
        if cur_key:
            if cur_key in res:
                res[cur_key].append(arg)
            else:
                res[cur_key] = [arg]
            limit -= 1
        else:
            flat.append(arg)
    return (flat, res)


def resolve_common_const(path):
    if path.startswith('${ARCADIA_ROOT}'):
        return path.replace('${ARCADIA_ROOT}', '$S', 1)
    if path.startswith('${ARCADIA_BUILD_ROOT}'):
        return path.replace('${ARCADIA_BUILD_ROOT}', '$B', 1)
    return path


def resolve_to_abs_path(path, source_root, build_root):
    if path.startswith('$S') and source_root is not None:
        return path.replace('$S', source_root, 1)
    if path.startswith('$B') and build_root is not None:
        return path.replace('$B', build_root, 1)
    return path


def resolve_to_ymake_path(path):
    return resolve_to_abs_path(path, '${ARCADIA_ROOT}', '${ARCADIA_BUILD_ROOT}')


def join_intl_paths(*args):
    return '/'.join(args)


def get(fun, num):
    return fun()[num][0]


def make_tuples(arg_list):
    def tpl():
        for x in arg_list:
            yield (x, [])

    return list(tpl())


def resolve_includes(unit, src, paths):
    return unit.resolve_include([src] + paths) if paths else []


def rootrel_arc_src(src, unit):
    if src.startswith('${ARCADIA_ROOT}/'):
        return src[16:]

    if src.startswith('${ARCADIA_BUILD_ROOT}/'):
        return src[22:]

    elif src.startswith('${CURDIR}/'):
        return unit.path()[3:] + '/' + src[10:]

    else:
        resolved = unit.resolve_arc_path(src)

        if resolved.startswith('$S/'):
            return resolved[3:]

        return src  # leave as is


def skip_build_root(x):
    if x.startswith('${ARCADIA_BUILD_ROOT}'):
        return x[len('${ARCADIA_BUILD_ROOT}'):].lstrip('/')

    return x


def get_interpreter_path():
    interpreter_path = [sys.executable]
    if 'ymake' in interpreter_path[0]:
        interpreter_path.append('--python')
    return interpreter_path


def filter_out_by_keyword(test_data, keyword):
    def _iterate():
        i = 0
        while i < len(test_data):
            if test_data[i] == keyword:
                i += 2
            else:
                yield test_data[i]
                i += 1

    return list(_iterate())


def generate_chunks(lst, chunk_size):
    for i in xrange(0, len(lst), chunk_size):
        yield lst[i:(i + chunk_size)]


def strip_roots(path):
    for prefix in ["$B/", "$S/"]:
        if path.startswith(prefix):
            return path[len(prefix):]
    return path
