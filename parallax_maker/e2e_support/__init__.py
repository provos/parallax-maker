"""Test-only support for deterministic browser tests.

Nothing in the production entry point imports this package.  The fakes are installed
only by :mod:`parallax_maker.e2e_server`, which must be invoked explicitly.
"""

from .fakes import INPAINT_PALETTES, install_fakes
from .fixtures import create_fixture_state, create_input_image

__all__ = [
    "INPAINT_PALETTES",
    "create_fixture_state",
    "create_input_image",
    "install_fakes",
]
