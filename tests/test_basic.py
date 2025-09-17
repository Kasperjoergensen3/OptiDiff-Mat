"""
Basic tests for the OptiDiff-Mat package
"""

import pytest
from optidiff_mat import __version__, __author__


def test_package_info():
    """Test that package metadata is accessible."""
    assert __version__ == "0.1.0"
    assert __author__ == "Kasperjoergensen3"


def test_package_import():
    """Test that the package can be imported."""
    import optidiff_mat
    assert optidiff_mat is not None


def test_submodules_import():
    """Test that submodules can be imported."""
    from optidiff_mat import models, data, utils
    assert models is not None
    assert data is not None
    assert utils is not None