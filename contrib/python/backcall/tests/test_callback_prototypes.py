import sys
from backcall import callback_prototype

@callback_prototype
def msg_callback(a, b, c, d=None, e=None, f=None):
    pass

def test_all_args():
    @msg_callback.adapt
    def thingy1(q, w, s, d, e, f):
        return q, w, s, d, e, f
    
    assert not getattr(thingy1, '__wrapped__', None)
    assert thingy1('A', 'B', 'C', d='D', e='E', f='F') == tuple('ABCDEF')

if sys.version_info[0] >= 3:
    exec("@msg_callback.adapt\n"
         "def thingy2(t, *, d=None):\n"
         "    return t, d")
    def test_some_args_kwonly():    
        assert getattr(thingy2, '__wrapped__', None)
        assert thingy2('A', 'B', 'C', d='D', e='E', f='F') == ('A', 'D')

def test_some_args_defaults():
    @msg_callback.adapt
    def thingy2b(t, d=None):
        return t, d
    
    assert getattr(thingy2b, '__wrapped__', None)
    assert thingy2b('A', 'B', 'C', d='D', e='E', f='F') == ('A', 'D')

def test_no_args():
    @msg_callback.adapt
    def thingy3():
        return 'Success'
    
    assert getattr(thingy3, '__wrapped__', None)
    assert thingy3('A', 'B', 'C', d='D', e='E', f='F') == 'Success'
