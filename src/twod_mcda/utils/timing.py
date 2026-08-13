"""Timing helper shared by algorithms and workflows."""

from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timer(name):
    """Measure and print the execution time of a code block."""
    tic = perf_counter()

    try:
        yield
    finally:
        elapsed = perf_counter() - tic
        print(f"[TIMING] {name:<70}: {elapsed:10.3f} s")
