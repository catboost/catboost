def test_initial_conftests(is_sessionstart_called_root, is_sessionstart_called_foo, is_sessionstart_called_bar):
    assert is_sessionstart_called_root
    assert is_sessionstart_called_foo
    assert not is_sessionstart_called_bar
