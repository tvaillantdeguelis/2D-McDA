import subprocess

def git_version():
    """
    Return a string describing the current Git state of the code.

    The function runs:
        git describe --tags --dirty --always

    Typical returned values:
        - "v1.4.2"                        (exactly on a tag)
        - "v1.4.2-5-g7a3f9c2"             (5 commits after tag v1.4.2)
        - "v1.4.2-5-g7a3f9c2-dirty"       (uncommitted local changes)
        - "g7a3f9c2"                      (no tags available)

    If the code is not inside a Git repository or Git is not available,
    the function returns None.
    """
    try:
        # Call Git to obtain a human-readable description of the current commit
        git_desc = subprocess.check_output(
            ["git", "describe", "--tags", "--dirty", "--always"],
            stderr=subprocess.DEVNULL
        )

        # Convert bytes to string and remove trailing newline
        return git_desc.decode().strip()

    except Exception:
        # Git is not available or the code is not in a Git repository
        return None


def get_full_version():
    """
    Return the best possible version string for this code.

    Priority order:
        1. Use the Git-based version string (tag + commit hash),
           which uniquely identifies the exact code state.
        2. If Git information is unavailable, fall back to the static
           Python version defined in __version__.

    Returned values examples:
        - "v1.4.2"
        - "v1.4.2-5-g7a3f9c2"
        - "v1.4.2-5-g7a3f9c2-dirty"
    """
    git_ver = git_version()

    if git_ver is not None:
        # Git information available: use it as the authoritative version
        return git_ver
    else:
        # Fallback: use the static version defined in the code
        return f"v{__version__}"