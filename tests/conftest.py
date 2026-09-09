"""Avoid BLAS thread oversubscription in small deterministic numerical tests."""

import pytest
from threadpoolctl import threadpool_limits


@pytest.fixture(scope="session", autouse=True)
def single_blas_thread():
    with threadpool_limits(limits=1):
        yield
