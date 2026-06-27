"""PyGithub wrapper: fetch issues, comments, and linked-PR diffs.

Exposes a single :class:`GitHubClient` whose :meth:`fetch_issues` returns typed
:class:`~rag.schemas.IssueRecord` objects ready for chunking. Concerns handled
here so the rest of the pipeline doesn't have to:

* Pull requests are filtered out (GitHub's issues API returns PRs too).
* Pagination is followed transparently.
* ``RateLimitExceededException`` is caught: we log the reset time, sleep until
  it passes, then continue. A small polite delay is inserted between requests.
* A linked PR (the merged PR that closed a closed issue) is found via the GraphQL
  ``closedByPullRequestsReferences`` field — GitHub does NOT surface these
  "linked pull request" links (closing keywords like *fixes #123*) in the REST
  timeline — with a REST timeline heuristic as fallback. Its per-file patches are
  concatenated into a diff string.

For tests, inject a fake PyGithub client via ``GitHubClient(client=...)`` — no
network is touched at construction time.

WRITE METHODS SAFETY NOTE: ``add_label`` / ``post_comment`` / ``open_draft_pr`` are
dumb building blocks — they just perform the GitHub call. They must NEVER be called
directly by agent nodes or anywhere else. The ONLY permitted caller is
``agent/guardrails.py``, which enforces the double dry-run gate, the allowlist, the
target-repo guard, and idempotency before any write happens.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from github import Auth, Github
from github.GithubException import RateLimitExceededException

from rag.schemas import IssueComment, IssueRecord
from shared.config import settings

logger = logging.getLogger("triage.github")

# Invisible marker appended to every agent-authored comment so we can detect our
# own prior comments and avoid double-posting (idempotency). Renders as nothing.
AGENT_COMMENT_MARKER = "<!-- triage-agent -->"


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
        # Token is kept for the GraphQL linked-PR lookup (the REST client can't
        # see closing-keyword PR links — see _find_linked_pr).
        self._token = token if token is not None else settings.github_token
        if client is not None:
            self._gh = client
        else:
            # An empty token still works for public repos (at a low rate limit).
            self._gh = (
                Github(auth=Auth.Token(self._token), per_page=100)
                if self._token
                else Github(per_page=100)
            )

    # -- public API --------------------------------------------------------
    def fetch_issues(
        self,
        repo: str,
        state: str = "all",
        limit: int | None = None,
        *,
        labels: list[str] | None = None,
        with_linked_pr: bool = True,
        with_comments: bool = True,
    ) -> list[IssueRecord]:
        """Fetch issues (excluding PRs) from ``owner/repo`` as typed records.

        ``labels`` (read-only) filters server-side to issues carrying ALL the
        given label names — used by the eval harness to pull priority-labelled
        issues cheaply (one page per label) instead of scanning the whole repo.
        ``with_linked_pr=False`` / ``with_comments=False`` skip the (expensive)
        linked-PR/diff resolution and comment pagination when the caller only
        needs title/body/labels (e.g. severity gold labels or a lean eval index).
        """
        repository = self._gh.get_repo(repo)
        kwargs: dict[str, Any] = {"state": state}
        if labels:
            # Resolve to Label objects so PyGithub builds the server-side filter
            # the same way across versions (avoids passing bare strings).
            kwargs["labels"] = [repository.get_label(name) for name in labels]
        records: list[IssueRecord] = []
        for issue in self._paginate(repository.get_issues(**kwargs)):
            # GitHub returns PRs through the issues API; skip them.
            if getattr(issue, "pull_request", None) is not None:
                continue
            records.append(
                self._build_record(
                    repository,
                    issue,
                    with_linked_pr=with_linked_pr,
                    with_comments=with_comments,
                )
            )
            if limit is not None and len(records) >= limit:
                break
        return records

    # -- record assembly ---------------------------------------------------
    def _build_record(
        self,
        repository: Any,
        issue: Any,
        *,
        with_linked_pr: bool = True,
        with_comments: bool = True,
    ) -> IssueRecord:
        labels = [label.name for label in issue.labels]
        comments = (
            [
                IssueComment(author=(c.user.login if c.user else None), body=c.body or "")
                for c in self._paginate(issue.get_comments())
            ]
            if with_comments
            else []
        )
        linked_pr, linked_pr_diff = (
            self._find_linked_pr(repository, issue) if with_linked_pr else (None, None)
        )
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
        """Find the merged PR that closed this issue and return (number, diff).

        Primary source is the GraphQL ``closedByPullRequestsReferences`` field —
        how GitHub records "linked pull requests" (closing keywords like
        *fixes #123*); these links do NOT appear in the REST timeline. We prefer a
        MERGED PR (it carries the actual fix). The REST timeline heuristic is a
        fallback. Any failure degrades to ``(None, None)`` so ingestion never
        crashes on one odd issue.
        """
        if issue.state != "closed":
            return None, None
        pr_number = self._linked_pr_via_graphql(repository, issue.number)
        if pr_number is None:
            pr_number = self._linked_pr_via_timeline(repository, issue)
        if pr_number is None:
            return None, None
        return pr_number, self._fetch_pr_diff(repository, pr_number)

    def _linked_pr_via_graphql(self, repository: Any, issue_number: int) -> int | None:
        """Return the number of the MERGED PR that closed the issue (GraphQL)."""
        if not self._token:
            return None
        full_name = getattr(repository, "full_name", None)
        if not full_name or "/" not in full_name:
            return None
        owner, name = full_name.split("/", 1)
        query = (
            "query($o:String!,$n:String!,$num:Int!){repository(owner:$o,name:$n){"
            "issue(number:$num){closedByPullRequestsReferences(first:10,includeClosedPrs:true)"
            "{nodes{number state}}}}}"
        )
        try:
            data = self._graphql(query, {"o": owner, "n": name, "num": issue_number})
        except Exception as exc:  # noqa: BLE001 — best-effort; fall back to timeline.
            logger.debug("GraphQL linked-PR lookup failed for #%s: %s", issue_number, exc)
            return None
        issue_node = ((data.get("data") or {}).get("repository") or {}).get("issue") or {}
        nodes = (issue_node.get("closedByPullRequestsReferences") or {}).get("nodes") or []
        merged = [n for n in nodes if n.get("state") == "MERGED"]
        return int(merged[0]["number"]) if merged else None

    def _graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Minimal GraphQL POST using the configured token (stdlib only)."""
        body = json.dumps({"query": query, "variables": variables}).encode()
        request = urllib.request.Request(
            "https://api.github.com/graphql",
            data=body,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "User-Agent": "triage-agent",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as resp:
            return json.load(resp)

    def _linked_pr_via_timeline(self, repository: Any, issue: Any) -> int | None:
        """Fallback: scan the REST timeline for a cross-referenced PR or closing commit."""
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
                    return pr_number
        except Exception as exc:  # noqa: BLE001 — best-effort.
            logger.debug("Timeline linked-PR lookup failed for #%s: %s", issue.number, exc)
        return None

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

    # -- idempotency reads (used by the guardrail before deciding to write) -
    def issue_has_label(self, repo: str, issue_number: int, label: str) -> bool:
        """True if ``label`` is already on the issue."""
        issue = self._gh.get_repo(repo).get_issue(issue_number)
        return any(lbl.name == label for lbl in issue.labels)

    def agent_comment_exists(
        self, repo: str, issue_number: int, marker: str = AGENT_COMMENT_MARKER
    ) -> bool:
        """True if the agent has already commented on the issue (detected by marker)."""
        issue = self._gh.get_repo(repo).get_issue(issue_number)
        return any(marker in (c.body or "") for c in self._paginate(issue.get_comments()))

    def branch_exists(self, repo: str, branch: str) -> bool:
        """True if ``branch`` already exists (so we don't open a duplicate draft PR)."""
        try:
            self._gh.get_repo(repo).get_branch(branch)
            return True
        except Exception:  # noqa: BLE001 — GithubException 404 -> not found.
            return False

    # -- WRITE methods — call ONLY via agent/guardrails.py -----------------
    def add_label(self, repo: str, issue_number: int, label: str) -> str:
        """Ensure the label exists in the repo, then add it to the issue. Returns issue URL."""
        repository = self._gh.get_repo(repo)
        self._ensure_label_exists(repository, label)
        issue = repository.get_issue(issue_number)
        issue.add_to_labels(label)  # additive — never removes existing labels
        return issue.html_url

    def post_comment(self, repo: str, issue_number: int, body: str) -> str:
        """Post an issue comment. Returns the comment URL."""
        issue = self._gh.get_repo(repo).get_issue(issue_number)
        return issue.create_comment(body).html_url

    def open_draft_pr(
        self,
        repo: str,
        issue_number: int,
        head_branch: str,
        title: str,
        body: str,
        patch: str,
        base: str | None = None,
    ) -> str:
        """Open a DRAFT PR carrying the proposed patch as an artifact. Returns PR URL.

        Design (deliberate, safe): we do NOT rewrite source files. We branch off
        the default branch, commit the proposed unified diff as a single artifact
        file ``.triage-agent/issue-<n>.diff`` **on the agent branch only** (the
        default branch is never touched), and open it as a *draft* PR whose body
        shows the patch in a fenced ``diff`` block with a "review/apply manually"
        note. So an ignored or closed PR leaves zero trace on ``main``, and nothing
        can be auto-merged into a broken state.
        """
        repository = self._gh.get_repo(repo)
        base_branch = base or repository.default_branch
        base_sha = repository.get_branch(base_branch).commit.sha

        # Create the agent branch off base (this is the only ref we create).
        repository.create_git_ref(ref=f"refs/heads/{head_branch}", sha=base_sha)
        self._throttle()

        # Commit the patch artifact — ON head_branch ONLY, never on base.
        repository.create_file(
            path=f".triage-agent/issue-{issue_number}.diff",
            message=f"triage-agent: proposed fix for #{issue_number} (draft, apply manually)",
            content=patch or "(empty patch)\n",
            branch=head_branch,
        )
        self._throttle()

        pull = repository.create_pull(
            title=title,
            body=self._draft_pr_body(issue_number, body, patch),
            head=head_branch,
            base=base_branch,
            draft=True,  # never ready-for-review; humans open/merge deliberately
        )
        return pull.html_url

    @staticmethod
    def _ensure_label_exists(repository: Any, label: str) -> None:
        try:
            repository.get_label(label)
        except Exception:  # noqa: BLE001 — not found; create it (best-effort).
            try:
                repository.create_label(name=label, color="ededed")
            except Exception as exc:  # noqa: BLE001 — race or perms; add_to_labels may still work.
                logger.debug("create_label(%s) failed: %s", label, exc)

    @staticmethod
    def _draft_pr_body(issue_number: int, body: str, patch: str) -> str:
        fence = "```"
        # Plain "#<n>" reference (a cross-link), NOT a closing keyword — merging
        # this artifact PR must never auto-close the issue.
        return (
            f"{body}\n\n"
            f"---\n"
            f"⚠️ **Proposed patch — review and apply manually.** Generated by the "
            f"triage agent for #{issue_number}. This draft PR intentionally does "
            f"**not** modify source files; the proposed change is shown below and "
            f"stored as `.triage-agent/issue-{issue_number}.diff` on this branch.\n\n"
            f"{fence}diff\n{patch}\n{fence}\n"
        )

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
