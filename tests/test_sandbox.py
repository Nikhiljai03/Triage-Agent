"""Sandbox runner tests — require a real Docker daemon.

These spin up real (locked-down) containers, so they are skipped cleanly when no
Docker daemon is reachable. To run them, ensure ``docker ps`` works on the host.
"""

from __future__ import annotations

import pytest

from sandbox.runner import SANDBOX_LABEL, SandboxRunner
from sandbox.schemas import ReproRequest


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


requires_docker = pytest.mark.skipif(
    not _docker_available(), reason="Docker daemon not available — sandbox tests need it"
)


@pytest.fixture(scope="module")
def runner() -> SandboxRunner:
    return SandboxRunner()


def _sandbox_containers() -> list:
    """All containers currently tagged as sandbox containers (for leak checks)."""
    import docker

    return docker.from_env().containers.list(all=True, filters={"label": f"{SANDBOX_LABEL}=1"})


@requires_docker
def test_failing_script_reproduces(runner: SandboxRunner) -> None:
    req = ReproRequest(
        files={"repro.py": "import sys\nprint('about to fail')\nraise ValueError('boom')\n"},
        command=["python", "repro.py"],
    )
    result = runner.run(req)
    assert result.reproduced is True
    assert result.exit_code not in (0, None)
    assert result.timed_out is False
    assert "boom" in result.stderr  # traceback goes to stderr


@requires_docker
def test_clean_script_does_not_reproduce(runner: SandboxRunner) -> None:
    req = ReproRequest(
        files={"ok.py": "print('all good')\n"},
        command=["python", "ok.py"],
    )
    result = runner.run(req)
    assert result.reproduced is False
    assert result.exit_code == 0
    assert "all good" in result.stdout


@requires_docker
def test_timeout_kills_runaway_loop(runner: SandboxRunner) -> None:
    import time

    req = ReproRequest(
        files={"loop.py": "while True:\n    pass\n"},
        command=["python", "loop.py"],
        timeout_seconds=3,
    )
    start = time.monotonic()
    result = runner.run(req)
    elapsed = time.monotonic() - start

    assert result.timed_out is True
    assert result.reproduced is False  # a hang is not a clean reproduction
    assert elapsed < 30  # returned promptly, didn't hang on the watchdog


@requires_docker
def test_network_is_blocked(runner: SandboxRunner) -> None:
    code = (
        "import urllib.request\n"
        "urllib.request.urlopen('http://example.com', timeout=5)\n"
        "print('REACHED_NETWORK')\n"
    )
    result = runner.run(ReproRequest(files={"net.py": code}, command=["python", "net.py"]))

    # No network -> the call fails (non-zero exit), and it never printed success.
    assert result.exit_code not in (0, None)
    assert "REACHED_NETWORK" not in result.stdout
    assert result.stderr  # an error/traceback was produced


@requires_docker
def test_no_leftover_container_after_run(runner: SandboxRunner) -> None:
    runner.run(ReproRequest(files={"ok.py": "print('hi')\n"}, command=["python", "ok.py"]))
    assert _sandbox_containers() == []  # removed in finally


@requires_docker
def test_cleanup_even_on_timeout(runner: SandboxRunner) -> None:
    runner.run(
        ReproRequest(
            files={"loop.py": "while True:\n    pass\n"},
            command=["python", "loop.py"],
            timeout_seconds=2,
        )
    )
    assert _sandbox_containers() == []  # killed AND removed despite the timeout
