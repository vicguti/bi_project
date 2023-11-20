import pytest

from functions.cleaning_functions import dummy_function


def test_dummy_function():
    value = 5
    try:
        dummy_function(value) == value * value
    except Exception as e:
        pytest.fail(f"Test failed: {e}")
