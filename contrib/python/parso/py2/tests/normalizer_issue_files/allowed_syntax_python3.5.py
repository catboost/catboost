"""
Mostly allowed syntax in Python 3.5.
"""


async def foo():
    await bar()
    #: E901
    yield from []
    return
    #: E901
    return ''


# With decorator it's a different statement.
@bla
async def foo():
    await bar()
    #: E901
    yield from []
    return
    #: E901
    return ''
