#!/usr/bin/env bash
# Start a Claude Code session powered by Meta's Muse Spark, kept apart from
# Claude's work (PROVENANCE.md, claude/adr/0004-muse-spark-as-claude-code-agent.md).
#
#   tools/muse-agent/start.sh            # run from anywhere; you start it, not Claude
#
# - works in its own git worktree on branch muse-agent, created from the tag
#   agent-era (the commit that registers claude/muse-agent-brief.md);
# - uses its own Claude Code config folder (MUSE_CONFIG_DIR, ~/.claude-muse),
#   so it sees none of Claude's history, memory or settings;
# - talks to Meta's Anthropic-compatible API (dev.meta.ai/docs/coding-agents),
#   with every model alias set to MUSE_MODEL (default muse-spark-1.3);
# - runs with tools/muse-agent/settings.json: Muse Spark co-author on commits,
#   writes only in muse-spark/, no .env, no push, sandboxed shell.
# The API key is read from .env inside this script and never printed.
set -euo pipefail

MAIN="$(cd "$(dirname "$0")/../.." && pwd)"
WT="${MUSE_WORKTREE:-$(dirname "$MAIN")/Hackathon-muse}"
MODEL="${MUSE_MODEL:-muse-spark-1.3}"
CONFIG="${MUSE_CONFIG_DIR:-$HOME/.claude-muse}"
PY="$MAIN/.venv/bin/python"

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

KEY="$("$PY" -c 'import sys; from dotenv import dotenv_values; sys.stdout.write(dotenv_values(".env").get("MODEL_API_KEY") or "")')"
[ -n "$KEY" ] || { echo "MODEL_API_KEY is not set in .env" >&2; exit 1; }
MUSE_SECRET_HASHES="$("$PY" tools/provenance.py secret-hashes)"

unset ANTHROPIC_API_KEY
export ANTHROPIC_BASE_URL="https://api.meta.ai"
export ANTHROPIC_AUTH_TOKEN="$KEY"
export ANTHROPIC_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
export CLAUDE_CODE_SUBAGENT_MODEL="$MODEL"
export ENABLE_TOOL_SEARCH="true"
export CLAUDE_CONFIG_DIR="$CONFIG"
export MUSE_SECRET_HASHES
unset KEY

cd "$WT"
echo "Muse Spark session: model $MODEL, worktree $WT, config $CONFIG"
echo "First prompt: Read claude/muse-agent-brief.md and do the task."
exec claude --settings "$MAIN/tools/muse-agent/settings.json" "$@"
