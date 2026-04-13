import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run tests that require a GPU",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="Pass --gpu to run GPU tests")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
