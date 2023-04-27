"""pytest configuration and fixtures"""

import sys

import pytest


@pytest.fixture(scope='session', autouse=True)
def win_py38_asyncio():
    """fix tornado compatibility on py38"""
    if sys.version_info < (3, 8) or not sys.platform.startswith('win'):
        return
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
