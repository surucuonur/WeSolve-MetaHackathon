"""Shared rules for keeping the Muse agent's work apart from Claude's (PROVENANCE.md).

The Muse agent is the Meta model working on muse-agent in a round: Muse Spark
in round 1, Muse Wonder Full from round 2 (claude/adr/0008-muse-wonder-full.md).
Used by the git hooks in tools/githooks/, tools/provenance_check.py,
tools/export_transcript.py and tools/muse-agent/start.sh.
"""

import os
import re
import subprocess
from pathlib import Path

MUSE_BRANCH = "muse-agent"
MUSE_DIR = "muse-spark/"
TRANSCRIPTS_DIR = MUSE_DIR + "transcripts/"
AGENT_ERA_TAG = "agent-era"
# muse-round-<N> marks round N's last commit; its files in muse-spark/ are then
# frozen (claude/adr/0009-rounds-build-on-each-other.md).
ROUND_TAG = "muse-round-"
AUTHOR = ("surucuonur", "25654558+surucuonur@users.noreply.github.com")
# Each Meta model that works on muse-agent, by model id prefix: the name its
# commit subjects start with, and its co-author line.
MUSE_AUTHORS = {
    "muse-spark-": (
        "Muse Spark",
        "Co-Authored-By: Muse Spark (Meta Model API) <muse-spark@meta-model-api.invalid>",
    ),
    "muse-wonder-full-": (
        "Muse Wonder Full",
        "Co-Authored-By: Muse Wonder Full (Muse Code) <muse-wonder-full@muse-code.invalid>",
    ),
}
SECRET_NAMES = ("MODEL_API_KEY", "HUGGINGFACE_API_KEY")
# In the main checkout; the second holds the key tools/muse-agent/start.sh uses.
ENV_FILES = (".env", "legal-agent-workshop/agent/.env")
COAUTHOR = re.compile(r"^co-authored-by:\s*(.+)$", re.I | re.M)


def git(*args, cwd=None):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, cwd=cwd
    ).stdout


def coauthors(message):
    return [m.strip() for m in COAUTHOR.findall(message)]


def is_muse(name):
    return name.lower().startswith(("muse spark", "muse wonder"))


def muse_author(model):
    """(name, co-author line) for a Meta model id, such as muse-spark-1.3-contributor."""
    for prefix, author in MUSE_AUTHORS.items():
        if model.startswith(prefix):
            return author
    raise ValueError(f"no Muse author for model {model!r}; add it to MUSE_AUTHORS")


def is_claude(name):
    low = name.lower()
    return "claude" in low or "anthropic.com" in low


def is_meta_model(model):
    """Whether a model id names one of Meta's Muse models, such as muse-spark-1.3."""
    return model.startswith("muse-")


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


def muse_worktree():
    """The checkout of muse-agent, found from any checkout of this repository."""
    for block in git("worktree", "list", "--porcelain").split("\n\n"):
        fields = dict(line.split(" ", 1) for line in block.splitlines() if " " in line)
        if fields.get("branch") == f"refs/heads/{MUSE_BRANCH}":
            return Path(fields["worktree"])
    raise RuntimeError(f"no checkout of {MUSE_BRANCH}; tools/muse-agent/start.sh creates it")


def finished_rounds(cwd=None):
    """The numbers of the rounds with a ROUND_TAG, in order."""
    tags = git("tag", "--list", ROUND_TAG + "*", cwd=cwd).split()
    return sorted(int(t[len(ROUND_TAG) :]) for t in tags if t[len(ROUND_TAG) :].isdigit())


def round_dir(n):
    """Where round n writes: round 1 at the top of muse-spark/, later rounds in round-<n>/."""
    return MUSE_DIR if n == 1 else f"{MUSE_DIR}round-{n}/"


def frozen_paths(cwd=None):
    """The files in muse-spark/ at any finished round's tag, which nothing may change."""
    paths = set()
    for n in finished_rounds(cwd):
        out = git("ls-tree", "-r", "-z", "--name-only", f"{ROUND_TAG}{n}", "--", MUSE_DIR, cwd=cwd)
        paths.update(filter(None, out.split("\0")))
    return paths


def muse_sessions_dir():
    """Where Muse Code keeps the Muse agent's session logs.

    tools/muse-agent/start.sh runs muse with HOME set to MUSE_HOME, so every
    session under this folder is a Muse agent session.
    """
    home = Path(os.environ.get("MUSE_HOME") or "~/.muse-spark").expanduser()
    return home / ".local/share/muse/sessions"


def secrets():
    """The API key values from the main checkout's ENV_FILES, or [] if none can be read.

    The values are only compared, never printed.
    """
    values = []
    for env_file in ENV_FILES:
        try:
            text = (main_checkout() / env_file).read_text()
        except OSError:
            continue
        for line in text.splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip().removeprefix("export ").strip() in SECRET_NAMES:
                values.append(value.strip().strip("'\""))
    return list(dict.fromkeys(v for v in values if len(v) >= 12))


def contains_secret(data, values):
    """Whether *data* (bytes) contains any of the secret *values*."""
    return any(v.encode() in data for v in values)
