import pytest

from make_fixture import CLASS_ORDER, make_fixture
from ears_ids.common.data import load_dataset


@pytest.fixture(scope="session")
def fixture_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("processed")
    make_fixture(root)
    return root


@pytest.fixture(scope="session")
def splits(fixture_root):
    return load_dataset("fixture", root=fixture_root)


@pytest.fixture(scope="session")
def class_order():
    return CLASS_ORDER
