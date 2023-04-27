"""0MQ Constants."""

# Copyright (c) PyZMQ Developers.
# Distributed under the terms of the Modified BSD License.

from zmq.backend import constants
from zmq.backend import has
from zmq.utils.constant_names import (
    base_names,
    switched_sockopt_names,
    int_sockopt_names,
    int64_sockopt_names,
    bytes_sockopt_names,
    fd_sockopt_names,
    ctx_opt_names,
    msg_opt_names,
)

#-----------------------------------------------------------------------------
# Python module level constants
#-----------------------------------------------------------------------------


__all__ = [
    'int_sockopts',
    'int64_sockopts',
    'bytes_sockopts',
    'ctx_opts',
    'ctx_opt_names',
    'DRAFT_API',
]

if constants.VERSION < 40200:
    DRAFT_API = False
else:
    DRAFT_API = bool(has('draft') and constants.DRAFT_API)

int_sockopts    = set()
int64_sockopts  = set()
bytes_sockopts  = set()
fd_sockopts     = set()
ctx_opts        = set()
msg_opts        = set()


if constants.VERSION < 30000:
    int64_sockopt_names.extend(switched_sockopt_names)
else:
    int_sockopt_names.extend(switched_sockopt_names)

_UNDEFINED = -9999


def _add_constant(name, container=None):
    """add a constant to be defined

    optionally add it to one of the sets for use in get/setopt checkers
    """
    c = getattr(constants, name, _UNDEFINED)
    if c == _UNDEFINED:
        return
    globals()[name] = c
    __all__.append(name)
    if container is not None:
        container.add(c)
    return c

for name in base_names:
    _add_constant(name)

for name in int_sockopt_names:
    _add_constant(name, int_sockopts)

for name in int64_sockopt_names:
    _add_constant(name, int64_sockopts)

for name in bytes_sockopt_names:
    _add_constant(name, bytes_sockopts)

for name in fd_sockopt_names:
    _add_constant(name, fd_sockopts)

for name in ctx_opt_names:
    _add_constant(name, ctx_opts)

for name in msg_opt_names:
    _add_constant(name, msg_opts)


# ensure some aliases are always defined
aliases = [
    ('DONTWAIT', 'NOBLOCK'),
    ('XREQ', 'DEALER'),
    ('XREP', 'ROUTER'),
]
for group in aliases:
    undefined = set()
    found = None
    for name in group:
        value = getattr(constants, name, -1)
        if value != -1:
            found = value
        else:
            undefined.add(name)
    if found is not None:
        for name in undefined:
            globals()[name] = found
            __all__.append(name)
