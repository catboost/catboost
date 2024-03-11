# Explicit type hints for dynamically-generated python file that mypy may not be able to find.

from __future__ import annotations

def build_config() -> dict[str, str]: ...
