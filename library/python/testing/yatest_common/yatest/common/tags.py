try:
    import pytest
    ya_external = getattr(pytest.mark, "ya:external")
except ImportError:
    ya_external = None
