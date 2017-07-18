# coding: utf-8


def extend_env_var(env, name, value, sep=":"):
    return sep.join(filter(None, [env.get(name), value]))
