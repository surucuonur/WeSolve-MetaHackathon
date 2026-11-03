"""Shared rules for keeping Muse Spark's work apart from Claude's (PROVENANCE.md).

Used by the git hooks in tools/githooks/, tools/provenance_check.py and
tools/export_transcript.py.
"""

import os
import re
import subprocess
from pathlib import Path

MUSE_BRANCH = "muse-agent"
MUSE_DIR = "muse-spark/"
AGENT_ERA_TAG = "agent-era"
AUTHOR = ("surucuonur", "25654558+surucuonur@users.noreply.github.com")
MUSE_COAUTHOR = (
    "Co-Authored-By: Muse Spark (Meta Model API) <muse-spark@meta-model-api.invalid>"
)
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
    return name.lower().startswith("muse spark")


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


def muse_sessions_dir():
    """Where Muse Code keeps the Muse Spark sessions' logs.

    tools/muse-agent/start.sh runs muse with HOME set to MUSE_HOME, so every
    session under this folder is a Muse Spark session.
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
