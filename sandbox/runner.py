"""Locked-down Docker sandbox for reproducing bugs from untrusted issue reports.

================================ THREAT MODEL ================================
The agent (Phase 4) will sometimes run code *derived from a bug report* to
confirm a crash reproduces. That code is UNTRUSTED — assume worst case. This
module is the ONLY place in the codebase permitted to execute issue-derived
code. Nothing runs on the host. Ever. (No subprocess/eval/exec of report
content anywhere else.)

What we defend against, and how:

* Untrusted code execution — the snippet may be malicious, not just buggy.
  -> It only ever runs inside a throwaway, locked-down container, never on the host.
* Data exfiltration / call-home / payload download.
  -> ``network_disabled=True``: the container has no network but loopback.
* Resource exhaustion / DoS (fork bombs, infinite loops, memory hogs).
  -> ``pids_limit`` (fork bombs), ``cpu_quota/period`` (busy loops),
     ``mem_limit`` + ``memswap_limit`` equal (no swap escape), and a hard
     wall-clock watchdog that kills the container on timeout.
* Host filesystem access / container escape.
  -> No host mounts; ``read_only`` root FS; ``cap_drop=["ALL"]``;
     ``no-new-privileges``; non-root ``user``. Only small size-limited tmpfs
     mounts are writable.
* Container lingering / resource leak.
  -> The container is ALWAYS killed and removed in a ``finally`` block, even on
     timeout or error.

Execution model: we start an idle container (``sleep``) with every lockdown
applied, stage the request files into its writable tmpfs work dir (base64 via
``exec`` — ``put_archive``/``docker cp`` is refused on a read-only rootfs), then
``exec_run`` the command. The hard timeout is enforced by a host-side watchdog
thread (we do NOT trust the workload to stop itself): on overrun we
``container.kill()`` and mark the run timed out.

Note: this is "Docker-out-of-Docker" — the runner talks to the HOST Docker
daemon to spawn sibling containers. A reachable Docker daemon is required.
=============================================================================
"""

from __future__ import annotations

import base64
import logging
import posixpath
import threading
import time

import docker
from docker.errors import DockerException, ImageNotFound, NotFound

from sandbox.policy import SandboxPolicy, default_policy
from sandbox.schemas import ReproRequest, ReproResult, judge_reproduced

logger = logging.getLogger("triage.sandbox")

# Label applied to every sandbox container so they are easy to find and reap.
SANDBOX_LABEL = "triage-sandbox"


class SandboxRunner:
    """Runs a :class:`ReproRequest` in a hardened, disposable Docker container."""

    def __init__(
        self, client: docker.DockerClient | None = None, policy: SandboxPolicy | None = None
    ) -> None:
        self._policy = policy or default_policy()
        try:
            self._client = client or docker.from_env()
            self._client.ping()
        except DockerException as exc:  # daemon down / socket missing
            raise RuntimeError(
                "Docker daemon is not reachable; the sandbox requires a running Docker daemon."
            ) from exc

    # ------------------------------------------------------------------ public
    def run(self, request: ReproRequest, policy: SandboxPolicy | None = None) -> ReproResult:
        """Execute ``request`` in a locked-down container and report the outcome."""
        policy = policy or self._policy
        timeout = request.timeout_seconds or policy.timeout_seconds
        cmd = self._command_argv(request.command)

        self._ensure_image(policy.image)

        container = None
        started = time.monotonic()
        timed_out = False
        exec_result: dict[str, object] = {}
        try:
            # Idle container with every lockdown applied (see threat model).
            container = self._client.containers.run(
                image=policy.image,
                command=["sleep", str(timeout + 30)],  # stays alive; we exec into it
                detach=True,
                network_disabled=policy.network_disabled,
                mem_limit=policy.mem_limit,
                memswap_limit=policy.memswap_limit,
                cpu_period=policy.cpu_period,
                cpu_quota=policy.cpu_quota,
                pids_limit=policy.pids_limit,
                read_only=policy.read_only,
                cap_drop=list(policy.cap_drop),
                security_opt=list(policy.security_opt),
                user=policy.user,
                working_dir=policy.work_dir,
                tmpfs=policy.tmpfs,
                labels={SANDBOX_LABEL: "1"},
                stdin_open=False,
                tty=False,
            )

            # Stage the files into the writable tmpfs work dir. put_archive /
            # docker cp is refused on a read-only rootfs, so we base64-decode each
            # file via exec into the tmpfs mount (which IS writable).
            self._stage_files(container, request.files, policy)

            # Run the command on a watchdog: kill the container if it overruns.
            worker = threading.Thread(
                target=self._exec, args=(container, cmd, policy, exec_result), daemon=True
            )
            worker.start()
            worker.join(timeout)
            if worker.is_alive():
                timed_out = True
                logger.warning("Sandbox run exceeded %ss; killing container.", timeout)
                self._safe_kill(container)
                worker.join(5)

            duration = time.monotonic() - started
            exit_code = None if timed_out else exec_result.get("exit_code")  # type: ignore[assignment]
            result = ReproResult(
                reproduced=judge_reproduced(exit_code, timed_out),
                stdout=self._truncate(exec_result.get("stdout"), policy.max_output_bytes),
                stderr=self._truncate(exec_result.get("stderr"), policy.max_output_bytes),
                exit_code=exit_code,
                timed_out=timed_out,
                duration_seconds=round(duration, 3),
                error=exec_result.get("error"),  # type: ignore[arg-type]
            )
            logger.info(
                "Sandbox run: image=%s cmd=%s duration=%.2fs exit=%s timed_out=%s reproduced=%s",
                policy.image,
                cmd,
                duration,
                exit_code,
                timed_out,
                result.reproduced,
            )
            return result
        except (DockerException, RuntimeError, ValueError) as exc:
            # Container/staging failure -> a runner-level error result, never a crash.
            logger.exception("Sandbox run failed")
            return ReproResult(
                reproduced=False,
                timed_out=timed_out,
                duration_seconds=round(time.monotonic() - started, 3),
                error=f"sandbox error: {exc}",
            )
        finally:
            # Guaranteed cleanup — never leak a container, even on timeout/error.
            self._safe_remove(container)

    # ----------------------------------------------------------------- helpers
    def _exec(
        self, container, cmd: list[str], policy: SandboxPolicy, out: dict[str, object]
    ) -> None:
        """Run the command inside the container (blocking; called in a thread)."""
        try:
            exit_code, streams = container.exec_run(
                cmd, demux=True, workdir=policy.work_dir, user=policy.user
            )
            stdout, stderr = streams if streams else (None, None)
            out["exit_code"] = exit_code
            out["stdout"] = stdout
            out["stderr"] = stderr
        except DockerException as exc:
            # Expected when the container is killed mid-exec (timeout path).
            out["error"] = f"exec error: {exc}"

    def _ensure_image(self, image: str) -> None:
        """Pull the image if it isn't present locally (with a log line)."""
        try:
            self._client.images.get(image)
        except ImageNotFound:
            logger.info("Sandbox image %s not present; pulling...", image)
            self._client.images.pull(image)

    @staticmethod
    def _command_argv(command: list[str] | str) -> list[str]:
        """Normalize the command: a bare string runs via ``sh -c``."""
        if isinstance(command, str):
            return ["/bin/sh", "-c", command]
        return list(command)

    def _stage_files(self, container, files: dict[str, str], policy: SandboxPolicy) -> None:
        """Write request files into the writable tmpfs work dir via a single exec.

        The base64 payload uses only ``[A-Za-z0-9+/=]`` so single-quoting in the
        shell is safe regardless of file contents. Filenames are validated to
        block path traversal out of the work dir.
        """
        lines = ["set -e"]
        for name, content in files.items():
            if name.startswith("/") or ".." in name.split("/"):
                raise ValueError(f"unsafe filename in repro request: {name!r}")
            target = posixpath.join(policy.work_dir, name)
            parent = posixpath.dirname(target)
            b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
            if parent and parent != policy.work_dir:
                lines.append(f"mkdir -p '{parent}'")
            lines.append(f"printf %s '{b64}' | base64 -d > '{target}'")
        exit_code, output = container.exec_run(
            ["/bin/sh", "-c", "\n".join(lines)], workdir=policy.work_dir, user=policy.user
        )
        if exit_code != 0:
            raise RuntimeError(f"failed to stage files (exit {exit_code}): {output!r}")

    @staticmethod
    def _truncate(data: object, limit: int) -> str:
        """Decode bytes (best-effort) and cap length so output can't blow up memory."""
        if not data:
            return ""
        raw = data if isinstance(data, bytes) else str(data).encode()
        if len(raw) > limit:
            return (
                raw[:limit].decode("utf-8", "replace")
                + f"\n...[truncated {len(raw) - limit} bytes]"
            )
        return raw.decode("utf-8", "replace")

    @staticmethod
    def _safe_kill(container) -> None:
        try:
            container.kill()
        except (DockerException, NotFound) as exc:
            logger.debug("kill() during timeout failed (already gone?): %s", exc)

    @staticmethod
    def _safe_remove(container) -> None:
        if container is None:
            return
        try:
            container.remove(force=True)
        except (DockerException, NotFound) as exc:
            # Log, but never mask the real result with a cleanup error.
            logger.warning(
                "Failed to remove sandbox container %s: %s", getattr(container, "id", "?"), exc
            )
