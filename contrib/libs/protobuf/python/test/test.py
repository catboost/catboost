def test_pyext_message_crash():
    # Since protobuf 3.2, a getter attribute on Message may crash the
    # interpreter.  We should ensure that it does not crash to support recursive
    # traversal of module attributes, in particular by the "astroid" package.
    # Upstream issue: https://github.com/google/protobuf/issues/2974
    from google.protobuf.pyext._message import Message
    Message._extensions_by_name
