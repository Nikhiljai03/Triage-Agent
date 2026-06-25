"""GitHub webhook signature verification and payload normalization.

Two pure-ish helpers used by the ``/webhook`` endpoint:

* :func:`verify_signature` — HMAC-SHA256 check against ``X-Hub-Signature-256``.
  **Fails closed**: with no secret configured, requests are rejected unless the
  explicit ``webhook_allow_unsigned`` dev flag is set.
* :func:`parse_issue_event` — normalize the raw payload into an
  :class:`IssueEvent`, returning ``None`` for events we intentionally ignore
  (non-opened/reopened actions, or pull requests).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any

from shared.config import settings
from shared.schemas import GitHubWebhookPayload, IssueEvent

logger = logging.getLogger("triage.webhook")

# Issue actions we actually triage. (GitHub sends many: edited, labeled, closed…)
ACTED_ACTIONS = frozenset({"opened", "reopened"})


def verify_signature(body: bytes, signature_header: str | None) -> bool:
    """Return True iff ``body`` matches ``X-Hub-Signature-256`` for the configured secret.

    Fails closed: no secret -> reject (unless ``settings.webhook_allow_unsigned``).
    """
    secret = settings.github_webhook_secret
    if not secret:
        if settings.webhook_allow_unsigned:
            logger.warning("No webhook secret set; allowing unsigned request (dev flag enabled).")
            return True
        logger.warning("No webhook secret set; rejecting request (fail closed).")
        return False

    if not signature_header:
        return False

    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    # Constant-time comparison to avoid timing attacks.
    return hmac.compare_digest(expected, signature_header)


def parse_issue_event(payload: dict[str, Any]) -> IssueEvent | None:
    """Normalize a raw GitHub issues payload, or return None if it should be ignored."""
    data = GitHubWebhookPayload.model_validate(payload)

    if data.action not in ACTED_ACTIONS:
        logger.info("Ignoring webhook with action=%r", data.action)
        return None

    issue = data.issue or {}
    # GitHub's issues feed can include PRs; an issue carrying "pull_request" is a PR.
    if "pull_request" in issue:
        logger.info("Ignoring pull-request event for #%s", issue.get("number"))
        return None

    labels = [
        label["name"]
        for label in issue.get("labels", []) or []
        if isinstance(label, dict) and label.get("name")
    ]
    return IssueEvent(
        repo=(data.repository or {}).get("full_name", ""),
        issue_number=int(issue.get("number", 0)),
        title=issue.get("title") or "",
        body=issue.get("body") or "",
        labels=labels,
        author=(issue.get("user") or {}).get("login"),
        action=data.action,
        html_url=issue.get("html_url") or "",
    )
