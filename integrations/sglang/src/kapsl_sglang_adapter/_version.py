"""The adapter's version, in one place.

It is part of the adapter profile a shared-pool registration claims, and a
managing runtime compares it with the version it certified. It equals the
distribution version in ``pyproject.toml``; ``tests/test_launch.py`` holds the
two together.
"""

ADAPTER_VERSION = "0.3.0"
