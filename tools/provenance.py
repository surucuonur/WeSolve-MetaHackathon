"""Shared rules for keeping Muse Spark's work apart from Claude's (PROVENANCE.md).

Used by the git hooks in tools/githooks/, tools/provenance_check.py and
tools/export_transcript.py. Run directly, it prints the secret fingerprints
the Muse Spark launcher hands to the hooks:

  .venv/bin/python tools/provenance.py secret-hashes
"""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

MUSE_BRANCH = "muse-agent"
MUSE_DIR = "muse-spark/"
AGENT_ERA_TAG = "agent-era"
AUTHOR = ("surucuonur", "25654558+surucuonur@users.noreply.github.com")
MUSE_COAUTHOR = (
    "Co-Authored-By: Muse Spark (Meta Model API) <muse-spark@meta-model-api.invalid>"
)
SECRET_NAMES = ("MODEL_API_KEY", "HUGGINGFACE_API_KEY")
COAUTHOR = re.compile(r"^co-authored-by:\s*(.+)$", re.I | re.M)


def git(*args, cwd=None):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, cwd=cwd
    ).stdout


def coauthors(message):
    return [m.strip() for m in COAUTHOR.findall(message)]


def is_muse(name):
    return name.lower().startswith("muse spark")


def is_claude(name):
    low = name.lower()
    return "claude" in low or "anthropic.com" in low


def kind(message):
    """'muse', 'claude', 'both' or 'human', from a commit message's co-authors."""
    names = coauthors(message)
    muse = any(is_muse(n) for n in names)
    claude = any(is_claude(n) for n in names)
    return (
        "both"
        if muse and claude
        else "muse" if muse else "claude" if claude else "human"
    )


def main_checkout():
    """The main checkout's root, also from inside a linked worktree."""
    common = Path(
        git("rev-parse", "--path-format=absolute", "--git-common-dir").strip()
    )
    return common.parent


def secrets():
    """The secret values from the main checkout's .env, or [] if it can't be read.

    The values are only compared, never printed.
    """
    try:
        text = (main_checkout() / ".env").read_text()
    except OSError:
        return []
    values = []
    for line in text.splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip().removeprefix("export ").strip() in SECRET_NAMES:
            value = value.strip().strip("'\"")
            if len(value) >= 12:
                values.append(value)
    return values


def secret_hashes():
    return [(len(v), hashlib.sha256(v.encode()).hexdigest()) for v in secrets()]


def parse_hashes(text):
    out = []
    for item in filter(None, (text or "").split(",")):
        n, _, h = item.partition(":")
        out.append((int(n), h))
    return out


def contains_secret(data, values=(), hashes=()):
    """Whether *data* (bytes) contains a secret, by value or by length:sha256."""
    if any(v.encode() in data for v in values):
        return True
    for n, h in hashes:
        # A key is one run of token characters, so only windows inside such runs.
        for m in re.finditer(rb"[A-Za-z0-9_\-.:+/=]{%d,}" % n, data):
            run = m.group()
            for i in range(len(run) - n + 1):
                if hashlib.sha256(run[i : i + n]).hexdigest() == h:
                    return True
    return False


if __name__ == "__main__":
    if sys.argv[1:] != ["secret-hashes"]:
        raise SystemExit("usage: provenance.py secret-hashes")
    print(",".join(f"{n}:{h}" for n, h in secret_hashes()))
