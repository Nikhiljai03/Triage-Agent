"""PyGithub wrapper: fetch issues, comments, and linked-PR diffs.

Exposes a single :class:`GitHubClient` whose :meth:`fetch_issues` returns typed
:class:`~rag.schemas.IssueRecord` objects ready for chunking. Concerns handled
here so the rest of the pipeline doesn't have to:

* Pull requests are filtered out (GitHub's issues API returns PRs too).
* Pagination is followed transparently.
* ``RateLimitExceededException`` is caught: we log the reset time, sleep until
  it passes, then continue. A small polite delay is inserted between requests.
* A linked PR (the one that closed a closed issue) is discovered via the issue
  timeline; its per-file patches are concatenated into a diff string.

For tests, inject a fake PyGithub client via ``GitHubClient(client=...)`` — no
network is touched at construction time.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from github import Auth, Github
from github.GithubException import RateLimitExceededException

from rag.schemas import IssueComment, IssueRecord
from shared.config import settings

logger = logging.getLogger("triage.github")


class GitHubClient:
    """Authenticated PyGithub wrapper that yields :class:`IssueRecord` objects."""

    def __init__(
        self,
        token: str | None = None,
        *,
        client: Github | None = None,
        request_delay: float = 0.1,
    ) -> None:
        self._request_delay = request_delay
        if client is not None:
            self._gh = client
        else:
            tok = token if token is not None else settings.github_token
            # An empty token still works for public repos (at a low rate limit).
            self._gh = Github(auth=Auth.Token(tok), per_page=100) if tok else Github(per_page=100)

    # -- public API --------------------------------------------------------
    def fetch_issues(
        self, repo: str, state: str = "all", limit: int | None = None
    ) -> list[IssueRecord]:
        """Fetch issues (excluding PRs) from ``owner/repo`` as typed records."""
        repository = self._gh.get_repo(repo)
        records: list[IssueRecord] = []
        for issue in self._paginate(repository.get_issues(state=state)):
            # GitHub returns PRs through the issues API; skip them.
            if getattr(issue, "pull_request", None) is not None:
                continue
            records.append(self._build_record(repository, issue))
            if limit is not None and len(records) >= limit:
                break
        return records

    # -- record assembly ---------------------------------------------------
    def _build_record(self, repository: Any, issue: Any) -> IssueRecord:
        labels = [label.name for label in issue.labels]
        comments = [
            IssueComment(author=(c.user.login if c.user else None), body=c.body or "")
            for c in self._paginate(issue.get_comments())
        ]
        linked_pr, linked_pr_diff = self._find_linked_pr(repository, issue)
        return IssueRecord(
            number=issue.number,
            title=issue.title or "",
            body=issue.body or "",
            state=issue.state,
            labels=labels,
            created_at=issue.created_at,
            comments=comments,
            linked_pr=linked_pr,
            linked_pr_diff=linked_pr_diff,
        )

    def _find_linked_pr(self, repository: Any, issue: Any) -> tuple[int | None, str | None]:
        """Best-effort: find the PR that closed this issue and return (number, diff).

        Only closed issues are inspected. We walk the timeline for a
        ``cross-referenced`` event whose source is a PR, or a ``closed`` event
        carrying a commit that belongs to a PR. Any failure degrades gracefully
        to ``(None, None)`` so ingestion never crashes on a single odd issue.
        """
        if issue.state != "closed":
            return None, None
        try:
            for event in self._paginate(issue.get_timeline()):
                etype = getattr(event, "event", None)
                if etype == "cross-referenced":
                    pr_number = self._pr_number_from_source(getattr(event, "source", None))
                elif etype == "closed":
                    pr_number = self._pr_for_commit(repository, getattr(event, "commit_id", None))
                else:
                    continue
                if pr_number is not None:
                    return pr_number, self._fetch_pr_diff(repository, pr_number)
        except Exception as exc:  # noqa: BLE001 — linked-PR detection is best-effort.
            logger.debug("Linked-PR lookup failed for issue #%s: %s", issue.number, exc)
        return None, None

    @staticmethod
    def _pr_number_from_source(source: Any) -> int | None:
        try:
            ref_issue = getattr(source, "issue", None)
            if ref_issue is not None and getattr(ref_issue, "pull_request", None) is not None:
                return int(ref_issue.number)
        except Exception:  # noqa: BLE001
            return None
        return None

    def _pr_for_commit(self, repository: Any, commit_sha: str | None) -> int | None:
        if not commit_sha:
            return None
        try:
            commit = repository.get_commit(commit_sha)
            self._throttle()
            for pull in commit.get_pulls():
                return int(pull.number)
        except Exception:  # noqa: BLE001
            return None
        return None

    def _fetch_pr_diff(self, repository: Any, pr_number: int) -> str | None:
        """Reconstruct a unified-diff-ish string from a PR's per-file patches."""
        try:
            pull = repository.get_pull(pr_number)
            self._throttle()
            parts: list[str] = []
            for f in self._paginate(pull.get_files()):
                header = f"diff --git a/{f.filename} b/{f.filename}"
                parts.append(f"{header}\n{f.patch or ''}")
            return "\n".join(parts) if parts else None
        except Exception as exc:  # noqa: BLE001
            logger.debug("PR diff fetch failed for #%s: %s", pr_number, exc)
            return None

    # -- pagination + rate limiting ---------------------------------------
    def _paginate(self, paginated: Any) -> Iterator[Any]:
        """Iterate a PyGithub PaginatedList, sleeping through rate limits.

        On ``RateLimitExceededException`` we sleep until the reset and retry the
        same page request (its internal cursor has not advanced). A small delay
        is inserted after each yielded item to stay polite.
        """
        iterator = iter(paginated)
        while True:
            try:
                item = next(iterator)
            except StopIteration:
                return
            except RateLimitExceededException:
                self._sleep_until_reset()
                continue
            self._throttle()
            yield item

    def _sleep_until_reset(self) -> None:
        reset = self._gh.get_rate_limit().core.reset  # UTC datetime
        seconds = max((reset - datetime.now(UTC)).total_seconds(), 0.0) + 1.0
        logger.warning(
            "GitHub rate limit hit; sleeping %.0fs until reset at %s.", seconds, reset.isoformat()
        )
        time.sleep(seconds)

    def _throttle(self) -> None:
        if self._request_delay:
            time.sleep(self._request_delay)
