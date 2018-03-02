from build.plugins import fortran


def test_include_parser():
    text = '''
c     %----------------------------------------------------%
c     | Include files for debugging and timing information |
c     %----------------------------------------------------%
c
      include   'debug.h'
      include   'stat.h'
c     include   'unused.h'
'''
    induced = list(fortran.FortranParser.parse_includes(text.split('\n')))
    assert induced == ['debug.h', 'stat.h']
