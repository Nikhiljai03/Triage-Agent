"""Post a realistic ``issues.opened`` webhook with a valid signature.

Demos the whole pipeline (webhook -> queue -> worker -> run store -> dashboard)
without a real GitHub webhook. Computes ``X-Hub-Signature-256`` from
``settings.github_webhook_secret`` so it passes the gateway's verification.

Usage::

    python -m scripts.simulate_webhook                 # uses settings.target_repo
    python -m scripts.simulate_webhook --title "Crash on startup" --number 4242
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import urllib.error
import urllib.request

from shared.config import settings


def build_payload(repo: str, number: int, title: str, body: str, action: str) -> dict:
    """Build a minimal-but-realistic GitHub issues webhook payload."""
    return {
        "action": action,
        "issue": {
            "number": number,
            "title": title,
            "body": body,
            "state": "open",
            "html_url": f"https://github.com/{repo}/issues/{number}",
            "labels": [{"name": "bug"}],
            "user": {"login": "octocat"},
        },
        "repository": {"full_name": repo},
        "sender": {"login": "octocat"},
    }


def sign(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post a signed issues.opened webhook.")
    parser.add_argument("--url", default="http://localhost:8000/webhook")
    parser.add_argument("--repo", default=settings.target_repo)
    parser.add_argument("--number", type=int, default=4242)
    parser.add_argument("--title", default="App crashes on startup with a null config")
    parser.add_argument("--body", default="Steps:\n1. Launch the app\n2. It crashes immediately.\n")
    parser.add_argument("--action", default="opened")
    args = parser.parse_args(argv)

    payload = build_payload(args.repo, args.number, args.title, args.body, args.action)
    body = json.dumps(payload).encode()

    headers = {"Content-Type": "application/json", "X-GitHub-Event": "issues"}
    if settings.github_webhook_secret:
        headers["X-Hub-Signature-256"] = sign(body, settings.github_webhook_secret)
    else:
        print(
            "WARNING: no GITHUB_WEBHOOK_SECRET set; sending unsigned (server will reject "
            "unless WEBHOOK_ALLOW_UNSIGNED=true)."
        )

    request = urllib.request.Request(args.url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request) as resp:
            print(f"HTTP {resp.status}: {resp.read().decode()}")
            return 0
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode()}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
