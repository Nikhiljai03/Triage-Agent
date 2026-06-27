"""Central security/resource policy for the sandbox — the single source of limits.

Every lockdown the runner applies is declared here and nowhere else, so the
security posture is auditable in one place. Each field is commented with *why*
it exists, tied back to the threat model in :mod:`sandbox.runner`.

Defaults come from :mod:`shared.config` (Phase 0 added ``sandbox_*`` settings) via
:func:`default_policy`, so limits are configurable per-deploy without code edits.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.config import settings

# CPU is capped as quota/period. period = 100ms; quota = fraction * period.
_CPU_PERIOD_US = 100_000


@dataclass(frozen=True)
class SandboxPolicy:
    """Immutable bundle of every sandbox limit. Construct via :func:`default_policy`."""

    # Base image the snippet runs in. Pinned/minimal to shrink attack surface.
    image: str = "python:3.11-slim"

    # Hard wall-clock kill — defends against infinite loops / hangs (DoS).
    timeout_seconds: int = 300

    # Memory cap. memswap_limit == mem_limit DISABLES swap, so a process can't
    # use swap to exceed the memory ceiling (resource exhaustion).
    mem_limit: str = "512m"
    memswap_limit: str = "512m"

    # CPU cap: quota microseconds of CPU per period. quota == period -> 1 core.
    # Caps compute so a busy loop can't starve the host (DoS).
    cpu_period: int = _CPU_PERIOD_US
    cpu_quota: int = _CPU_PERIOD_US

    # Max number of processes/threads — stops fork bombs (DoS).
    pids_limit: int = 128

    # No network at all — blocks data exfiltration, phone-home, payload download.
    network_disabled: bool = True

    # Read-only root filesystem — code can't tamper with the image or persist
    # anything outside the explicit writable tmpfs mounts (host/escape defense).
    read_only: bool = True

    # Drop every Linux capability — no raw sockets, no mounts, no ptrace, etc.
    cap_drop: tuple[str, ...] = ("ALL",)

    # Prevent setuid/privilege escalation inside the container.
    security_opt: tuple[str, ...] = ("no-new-privileges:true",)

    # Run as an unprivileged, non-root UID:GID (defense in depth vs. escape).
    user: str = "1000:1000"

    # Writable scratch space. Root FS is read-only, so the code needs *somewhere*
    # to write — small, size-limited tmpfs (RAM-backed) mounted at the work dir
    # and /tmp. Size cap prevents filling host memory.
    work_dir: str = "/sandbox"
    tmpfs_size: str = "64m"

    # Cap captured stdout/stderr so a noisy/abusive program can't blow up the
    # runner's memory; output beyond this is truncated with a marker.
    max_output_bytes: int = 64 * 1024

    @property
    def tmpfs(self) -> dict[str, str]:
        """tmpfs mount spec for the work dir and /tmp (RAM-backed, size-limited)."""
        opts = f"size={self.tmpfs_size},mode=1777"
        return {self.work_dir: opts, "/tmp": opts}


def default_policy() -> SandboxPolicy:
    """Build the default policy, sourcing limits from ``settings`` where sensible."""
    cpu_quota = max(int(settings.sandbox_cpu_quota * _CPU_PERIOD_US), 1_000)
    return SandboxPolicy(
        image=settings.sandbox_image,
        timeout_seconds=settings.sandbox_timeout_seconds,
        mem_limit=settings.sandbox_mem_limit,
        memswap_limit=settings.sandbox_mem_limit,  # == mem_limit -> swap disabled
        cpu_period=_CPU_PERIOD_US,
        cpu_quota=cpu_quota,
    )
