import threading

from catboost.core import _CustomLoggersStack


def test_custom_loggers_stack_push_pop_balance():
    # A single-threaded push/pop pair, including reentrancy, must be balanced
    # and must not raise.
    stack = _CustomLoggersStack()
    stack.push()
    stack.push()
    stack.pop()
    stack.pop()


def test_custom_loggers_stack_pop_on_empty_is_noop():
    # Under concurrent fit() from multiple threads, push() can be a no-op
    # (it returns early when another thread owns the stack), so push/pop are
    # not guaranteed to be balanced. Popping an empty stack must be a no-op
    # rather than raising, otherwise parallel fit() crashes with
    # "RuntimeError: Attempt to pop from an empty stack" (see #2620).
    stack = _CustomLoggersStack()
    stack.push()
    stack.pop()

    # Reproduce the state that triggers the crash: the current thread is
    # recorded as the owning thread, but the stack is empty.
    stack._owning_thread_id = threading.current_thread().ident
    stack._stack = []

    stack.pop()
