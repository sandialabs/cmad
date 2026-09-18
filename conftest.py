"""Test collection options: slow tests run only with --include-slow."""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--include-slow", action="store_true", default=False,
        help="run the tests marked slow as well",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: skipped unless pytest runs with --include-slow",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--include-slow") or config.option.markexpr:
        return
    skip = pytest.mark.skip(reason="slow; run with --include-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
