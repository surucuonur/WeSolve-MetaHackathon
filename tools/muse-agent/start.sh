#!/usr/bin/env bash
# Start a Muse Spark session: a Claude Code session powered by Meta's Muse Spark,
# kept apart from Claude's work (PROVENANCE.md, claude/adr/0004-muse-spark-as-claude-code-agent.md,
# claude/adr/0005-meta-settings-in-muse-folder-contributor-tier.md).
#
#   tools/muse-agent/start.sh [claude args]   # run from anywhere; you start it, not Claude
#
# This script is the only way to start one. It
# - checks that the session's own Claude Code config folder (MUSE_CONFIG_DIR,
#   ~/.claude-muse) points at Meta: its settings.json must set
#   env.ANTHROPIC_BASE_URL to https://api.meta.ai and env.ANTHROPIC_MODEL to a
#   muse-spark model. surucuonur writes those settings, with the API key; this
#   script reads only those two fields;
# - works in its own git worktree on branch muse-agent, created from the tag
#   agent-era (the commit that registers claude/muse-agent-brief.md);
# - uses that config folder, so it sees none of Claude's history, memory or settings;
# - sets every model alias to the folder's model (dev.meta.ai/docs/coding-agents);
# - runs with tools/muse-agent/settings.json: Muse Spark co-author on commits,
#   writes only in muse-spark/, no .env, no push.
#
# When the session ends, in the worktree:
#   .venv/bin/python tools/export_transcript.py
#   git add muse-spark/transcripts && git commit -m "Muse Spark: session transcripts" \
#     -m "Co-Authored-By: Muse Spark (Meta Model API) <muse-spark@meta-model-api.invalid>"
set -euo pipefail

MAIN="$(cd "$(dirname "$0")/../.." && pwd)"
WT="${MUSE_WORKTREE:-$(dirname "$MAIN")/Hackathon-muse}"
CONFIG="${MUSE_CONFIG_DIR:-$HOME/.claude-muse}"
PY="$MAIN/.venv/bin/python"

MODEL="$("$PY" - "$CONFIG/settings.json" <<'EOF'
import json
import sys

path = sys.argv[1]
try:
    data = json.load(open(path))
except (OSError, ValueError) as e:
    sys.exit(f"can't read {path} ({e.__class__.__name__})")
env = data.get("env") if isinstance(data, dict) else None
env = env if isinstance(env, dict) else {}
url = str(env.get("ANTHROPIC_BASE_URL", "")).rstrip("/")
model = str(env.get("ANTHROPIC_MODEL", ""))
if url != "https://api.meta.ai" or not model.startswith("muse-spark"):
    sys.exit(
        f"{path} must set env.ANTHROPIC_BASE_URL to https://api.meta.ai "
        "and env.ANTHROPIC_MODEL to a muse-spark model"
    )
print(model)
EOF
)"

cd "$MAIN"
git rev-parse -q --verify refs/tags/agent-era >/dev/null \
  || { echo "tag agent-era is missing: the brief must be registered first" >&2; exit 1; }
[ "$(git config core.hooksPath)" = "$MAIN/tools/githooks" ] \
  || { echo "hooks aren't enabled: git config core.hooksPath $MAIN/tools/githooks" >&2; exit 1; }

if [ ! -d "$WT" ]; then
  if git rev-parse -q --verify refs/heads/muse-agent >/dev/null; then
    git worktree add "$WT" muse-agent
  else
    git worktree add -b muse-agent "$WT" agent-era
  fi
  git -C "$WT" submodule update --init --reference "$MAIN/third_party/mlx-lm" third_party/mlx-lm
fi
ln -sfn "$MAIN/.venv" "$WT/.venv"
ln -sfn "$MAIN/models" "$WT/models"
[ "$(git -C "$WT" branch --show-current)" = "muse-agent" ] \
  || { echo "$WT is not on branch muse-agent" >&2; exit 1; }

# Only the folder's settings supply the API; no other key reaches the session.
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL
export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
export CLAUDE_CODE_SUBAGENT_MODEL="$MODEL"
export CLAUDE_CODE_AUTO_MODE_MODEL="$MODEL"
export ENABLE_TOOL_SEARCH="true"
export CLAUDE_CONFIG_DIR="$CONFIG"
export MUSE_CONFIG_DIR="$CONFIG"

cd "$WT"
echo "Muse Spark session: model $MODEL, worktree $WT, config $CONFIG"
echo "First prompt: Read claude/muse-agent-brief.md and do the task."
exec claude --settings "$MAIN/tools/muse-agent/settings.json" "$@"
