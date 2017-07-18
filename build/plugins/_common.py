import sys
import hashlib


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
    return hashlib.md5(path).hexdigest()


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


def join_intl_paths(*args):
    return '/'.join(args)


def get(fun, num):
    return fun()[num][0]


def make_tuples(arg_list):
    def tpl():
        for x in arg_list:
            yield (x, [])

    return list(tpl())


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


# ----------------_common tests------------------ #
def test_sort_by_keywords():
    keywords = {'KEY1': 2, 'KEY2': 0, 'KEY3': 1}
    args = 'aaaa bbbb KEY2 KEY1 kkk10 kkk11 ccc ddd KEY3 kkk3 eee'.split()
    flat, spec = sort_by_keywords(keywords, args)
    assert flat == ['aaaa', 'bbbb', 'ccc', 'ddd', 'eee']
    assert spec == {'KEY1': ['kkk10', 'kkk11'], 'KEY2': True, 'KEY3': ['kkk3']}

    keywords = {'KEY1': 0, 'KEY2': 4}
    args = 'aaaa KEY2 eee'.split()
    flat, spec = sort_by_keywords(keywords, args)
    assert flat == ['aaaa']
    assert spec == {'KEY2': ['eee']}

    keywords = {'KEY1': 2, 'KEY2': 2}
    args = 'KEY1 k10 KEY2 k20 KEY1 k11 KEY2 k21 KEY1 k13'.split()
    flat, spec = sort_by_keywords(keywords, args)
    assert flat == []
    assert spec == {'KEY1': ['k10', 'k11', 'k13'], 'KEY2': ['k20', 'k21']}
