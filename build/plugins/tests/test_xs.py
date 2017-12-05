from build.plugins import xs


def test_include_parser():
    text = '''
aaaaa
#include <induced1>
# include <not_induced>
#include<induced2>
INCLUDE: included1
INCLUDE : included2
INCLUDE: included3  // asdasd
//INCLUDE : not_included
'''
    includes, induced = xs.XsParser.parse_includes(text.split('\n'))
    assert includes == ['included1', 'included2', 'included3', ]
    assert induced == ['induced1', 'induced2']
