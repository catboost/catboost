import os

from ymake import subst
from _common import lazy


@lazy
def get_file():
    return open(subst(os.path.join('$B', 'metaquery.lst')), 'w')


def onmetaqueryfiles(unit, *args):
    """
        @usage: METAQUERYFILES(filenames...)

        Deprecated
    """
    f = get_file()
    f.write('####\n')
    f.write('unit-name: {}\n'.format(unit.name()))
    f.write('unit-path: {}\n'.format(unit.path()))
    f.write('args: {}\n'.format(', '.join(args)))
