from build.plugins import nlg


def test_output_paths():
    expected = ['register.cpp', 'register.h', 'foo.nlg.cpp', 'foo.nlg.h']
    assert expected == nlg.output_paths(['foo.nlg'])
