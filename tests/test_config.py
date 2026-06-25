"""Smoke test: the central config imports and defaults to safe values.

Proves the Phase 0 plumbing — pydantic-settings loads without a populated
``.env`` and the safety defaults are correct before any real logic exists.
"""

from shared.config import settings


def test_config_loads_with_safe_defaults() -> None:
    """Importing settings must succeed and default to safe dry-run mode."""
    assert settings.dry_run is True
    assert settings.enable_live_writes is False
