import pytest
import os

def pytest_addoption(parser):
    parser.addoption(
        "--data_dir",
        action="store",
        default=os.environ.get("DATA_DIR", None),
        help="Path to external data directory"
    )

@pytest.fixture
def data_dir(request):
    path = request.config.getoption("--data_dir")
    if path is None:
        pytest.skip("No data_dir provided")
    if not os.path.exists(path):
        pytest.fail(f"Data directory does not exist: {path}")
    return path
