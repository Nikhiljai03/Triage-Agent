"""Typed request/result models for the sandbox runner."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReproRequest(BaseModel):
    """A request to reproduce a bug in the sandbox.

    ``files`` maps *relative* filenames to their text contents; they are dropped
    into the container's writable work dir (e.g. ``{"repro.py": "...code..."}``).
    ``command`` is what to run there (a list like ``["python", "repro.py"]`` is
    run directly; a bare string is run via ``sh -c``). ``timeout_seconds`` overrides
    the policy's hard timeout for this one run when set.
    """

    files: dict[str, str]
    command: list[str] | str
    timeout_seconds: int | None = None


class ReproResult(BaseModel):
    """The outcome of one sandbox run."""

    reproduced: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    duration_seconds: float = 0.0
    error: str | None = Field(default=None, description="runner-level error, not program output")


def judge_reproduced(exit_code: int | None, timed_out: bool) -> bool:
    """Decide whether a bug *reproduced*.

    Rule (explicit and intentionally simple; override in a later phase if needed):
    a bug reproduces when the program **failed** — i.e. it exited with a non-zero
    code — and did **not** time out. A timeout is reported separately
    (``timed_out=True``) and is NOT counted as a clean reproduction, because a
    hang tells us nothing definitive about the reported crash.
    """
    return (not timed_out) and exit_code is not None and exit_code != 0
