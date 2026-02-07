import sys


def _numba_status():
    if "numba" not in sys.modules:
        return "MISSING"
    m = sys.modules["numba"]
    return f"present type={type(m).__name__} has_core={hasattr(m, 'core')}"


def pytest_runtest_teardown(item, nextitem):
    # Print test name and numba status after the test has run
    try:
        status = _numba_status()
    except Exception as e:
        status = f"ERROR: {e}"
    print(f"[NUMBA-STATUS] after {item.nodeid}: {status}")
