"""Timing helper shared by algorithms and workflows.

This timer prints entry messages and, on exit, prints the elapsed time
on a separate line aligned with the entry message. Nested timers are
indented by four spaces per nesting level.
"""

from contextlib import contextmanager
from time import perf_counter
import threading


# Thread-local state so concurrent threads have independent depths
_timer_state = threading.local()


@contextmanager
def timer(name, enabled=True):
    """Measure and print the execution time of a code block.

    Usage:
        with timer("Some step..."):
            ...

        with timer("Temporarily hidden step...", enabled=False):
            ...

    This will print the step message on enter and the elapsed time on
    exit on its own line. Nested timers increase the indentation level.
    When disabled, the wrapped block runs without timing or output.
    """
    if not enabled:
        yield
        return

    # Capture current depth and compute indent for consistency.
    depth = getattr(_timer_state, "depth", 0)
    indent = " " * (4 * depth)

    # Entry message
    print(f"{indent}{name}...")

    # Increase depth for nested timers
    _timer_state.depth = depth + 1
    tic = perf_counter()

    try:
        yield
    finally:
        elapsed = perf_counter() - tic
        # Restore the original depth (use the saved local `depth` to avoid
        # relying on potentially-modified thread-local state).
        _timer_state.depth = depth
        indent = " " * (4 * depth)

        # Print elapsed time on its own line, then a blank line for readability
        print(f"{indent}{elapsed:.3f} s\n")
