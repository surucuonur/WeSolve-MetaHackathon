#!/usr/bin/env bash
# Start a Muse Spark session in Muse Code, Meta's own coding agent, kept apart
# from Claude's work (PROVENANCE.md, claude/adr/0006-muse-spark-in-muse-code.md).
#
#   tools/muse-agent/start.sh [exec] [muse flags]   # run from anywhere; you start it, Claude only when asked
#
# The session's first prompt is fixed here (PROMPT): run the round skill,
# .agents/skills/muse-round/SKILL.md. MUSE_PROMPT replaces it for a one-off
# session such as a smoke test. Extra flags go to muse before it,
# e.g. --reasoning-effort xhigh. With exec first, the session runs headless
# (muse exec) instead of in the terminal UI, e.g. for a smoke test.
#
# This script is the only way to start one. It
# - takes the API key (MODEL_API_KEY) from legal-agent-workshop/agent/.env and
#   gives it only to muse, as META_API_KEY; it never prints or stores it;
# - works in its own git worktree on branch muse-agent, created from the tag
#   muse-code-era (the commit that registers the skill and ADR 0006);
# - fast-forwards muse-agent to main when main already contains it (after a
#   round's merge), so the session sees Claude's newest findings and grader;
# - runs MUSE_MODEL (default muse-spark-1.3-contributor);
# - trusts the worktree for this run, which loads its skill and lets the lead
#   start helper agents, and lets 12 agents run at once (default 8), enough
#   for the round's swarm of explorers, a verifier and the lead;
# - gives muse a home folder of its own, MUSE_HOME (default ~/.muse-spark), so
#   it finds none of the personal skills and rules under the real home
#   (~/.agents/skills, ~/.claude/skills, ~/.claude/CLAUDE.md), which it would
#   otherwise load, and keeps its settings and session logs there. Muse Code's
#   settings for these (context.foreign_personal_*) don't stop them, and
#   ~/.agents/skills counts as its own;
# - turns Muse Code's OS sandbox off: it keeps .git read-only, and a worktree
#   commits into the main checkout's .git. Tool approvals stay on.
#
# When the session ends, in the worktree:
#   .venv/bin/python tools/export_transcript.py
#   git add muse-spark/transcripts && git commit -m "Muse Spark: session transcripts" \
#     -m "Co-Authored-By: Muse Spark (Meta Model API) <muse-spark@meta-model-api.invalid>"
set -euo pipefail

MAIN="$(cd "$(dirname "$0")/../.." && pwd)"
WT="${MUSE_WORKTREE:-$(dirname "$MAIN")/Hackathon-muse}"
MODEL="${MUSE_MODEL:-muse-spark-1.3-contributor}"
MUSE_HOME="${MUSE_HOME:-$HOME/.muse-spark}"  # also read by tools/provenance.py
TAG="muse-code-era"
ENV_FILE="$MAIN/legal-agent-workshop/agent/.env"  # also scanned by tools/provenance.py
PY="$MAIN/.venv/bin/python"
PROMPT="${MUSE_PROMPT:-Run the muse-round skill: read .agents/skills/muse-round/SKILL.md and follow it.}"
SUB=""
if [ "${1:-}" = exec ]; then SUB=exec; shift; fi

command -v muse >/dev/null \
  || { echo "Muse Code isn't installed (https://dev.meta.ai/docs/muse-code)" >&2; exit 1; }
KEY="$("$PY" -c 'import sys; from dotenv import dotenv_values; sys.stdout.write(dotenv_values(sys.argv[1]).get("MODEL_API_KEY") or "")' "$ENV_FILE")"
[ -n "$KEY" ] || { echo "MODEL_API_KEY is not set in $ENV_FILE" >&2; exit 1; }

cd "$MAIN"
git rev-parse -q --verify "refs/tags/$TAG" >/dev/null \
  || { echo "tag $TAG is missing: the skill and ADR 0006 must be registered first" >&2; exit 1; }
[ "$(git config core.hooksPath)" = "$MAIN/tools/githooks" ] \
  || { echo "hooks aren't enabled: git config core.hooksPath $MAIN/tools/githooks" >&2; exit 1; }

if [ ! -d "$WT" ]; then
  if git rev-parse -q --verify refs/heads/muse-agent >/dev/null; then
    git worktree add "$WT" muse-agent
  else
    git worktree add -b muse-agent "$WT" "$TAG"
  fi
  git -C "$WT" submodule update --init --reference "$MAIN/third_party/mlx-lm" third_party/mlx-lm
fi
ln -sfn "$MAIN/.venv" "$WT/.venv"
ln -sfn "$MAIN/models" "$WT/models"
[ "$(git -C "$WT" branch --show-current)" = "muse-agent" ] \
  || { echo "$WT is not on branch muse-agent" >&2; exit 1; }

if git merge-base --is-ancestor muse-agent main; then
  git -C "$WT" merge --ff-only --quiet main
elif ! git merge-base --is-ancestor main muse-agent; then
  echo "note: main has commits muse-agent lacks; merge muse-agent into main first so the session sees them" >&2
fi

unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL
export META_API_KEY="$KEY"
unset KEY

cd "$WT"
echo "Muse Spark session in Muse Code: model $MODEL, worktree $WT, branch muse-agent at $(git rev-parse --short HEAD), home $MUSE_HOME"
echo "First prompt: $PROMPT"
mkdir -p "$MUSE_HOME"
export HOME="$MUSE_HOME"
# Muse Code would look for its config, data and other agents' folders here first.
unset XDG_CONFIG_HOME XDG_DATA_HOME XDG_STATE_HOME XDG_CACHE_HOME CLAUDE_CONFIG_DIR CODEX_HOME
exec muse ${SUB:+"$SUB"} --model "$MODEL" --workspace "$WT" --trust-workspace --disable-sandbox \
  -c agents.execution_capacity=12 "$@" "$PROMPT"
