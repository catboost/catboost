import os

from _common import sort_by_keywords


def onstyle(unit, *args):
    keywords = {'FOLDER': 1}

    flat_args, spec_args = sort_by_keywords(keywords, args)

    def it():
        yield 'DONT_PARSE'

        for f in flat_args:
            yield spec_args['FOLDER'][0] + '/' + f
            yield '/cpp_style/' + f

    unit.onresource(list(it()))
