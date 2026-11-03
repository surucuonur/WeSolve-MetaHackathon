#!/usr/bin/env bash
# Start a Muse Spark session in Muse Code, Meta's own coding agent, kept apart
# from Claude's work (PROVENANCE.md, claude/adr/0006-muse-spark-in-muse-code.md).
#
#   tools/muse-agent/start.sh [muse flags]   # run from anywhere; you start it, not Claude
#
# The session's first prompt is fixed here (PROMPT): run the round skill,
# .agents/skills/muse-round/SKILL.md. Extra flags go to muse before it,
# e.g. --reasoning-effort xhigh.
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
#   start helper agents, and leaves out personal skills and rules from outside
#   the repository (~/.claude/skills, ~/.agents/skills, ~/.claude/CLAUDE.md);
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
TAG="muse-code-era"
ENV_FILE="$MAIN/legal-agent-workshop/agent/.env"  # also scanned by tools/provenance.py
PY="$MAIN/.venv/bin/python"
PROMPT="Run the muse-round skill: read .agents/skills/muse-round/SKILL.md and follow it."

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
echo "Muse Spark session in Muse Code: model $MODEL, worktree $WT, branch muse-agent at $(git rev-parse --short HEAD)"
echo "First prompt: $PROMPT"
exec muse --model "$MODEL" --workspace "$WT" --trust-workspace --disable-sandbox \
  -c context.foreign_personal_skills=false -c context.foreign_personal_rules=false \
  "$@" "$PROMPT"
