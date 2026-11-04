#!/usr/bin/env bash
# Start a Muse agent session in Muse Code, Meta's own coding agent, kept apart
# from Claude's work (PROVENANCE.md, claude/adr/0006-muse-spark-in-muse-code.md).
#
#   tools/muse-agent/start.sh [exec] [muse flags]   # run from anywhere; you start it, Claude only when asked
#   MUSE_AUTH=account tools/muse-agent/start.sh login   # only for a subscription model, once
#
# The session's first prompt is fixed here (PROMPT): run the round skill,
# .agents/skills/muse-round/SKILL.md, as the model it names, with the co-author
# line its commits end with. MUSE_PROMPT replaces it for a one-off session such
# as a smoke test. Extra flags go to muse before it. With exec first, the
# session runs headless (muse exec) instead of in the terminal UI.
#
# This script is the only way to start one. It
# - runs MUSE_MODEL (default muse-wonder-1.0, Muse Wonder, for the last round,
#   as in round 2; ADR 0014; round 3 ran muse-spark-1.3-contributor, ADR 0012)
#   at reasoning effort MUSE_EFFORT (default max), both passed explicitly so
#   the session log records them;
# - signs in as MUSE_AUTH says:
#   - key (default): the pay-as-you-go API key (MODEL_API_KEY) from
#     legal-agent-workshop/agent/.env, given only to muse, as META_API_KEY; it
#     is never printed or stored. Every round so far ran this way;
#   - account: a Muse Code subscription, which Muse Wonder Full
#     (muse-wonder-full-1.0) needs. META_API_KEY is unset, because an API key
#     always takes priority over the account login. Sign in once with
#     `MUSE_AUTH=account start.sh login`. Contributor mode MUSE_CONTRIBUTOR
#     (default on) is a subscription setting, passed only here;
# - works in its own git worktree on branch muse-agent, created from the tag
#   muse-code-era (the commit that registers the skill and ADR 0006);
# - fast-forwards muse-agent to main when main already contains it (after a
#   round's merge), so the session sees Claude's newest findings and grader;
# - trusts the worktree for this run, which loads its skill and lets the lead
#   start helper agents, and lets 12 agents run at once (default 8), enough
#   for the round's swarm of explorers, a verifier and the lead;
# - gives muse a home folder of its own, MUSE_HOME (default ~/.muse-spark), so
#   it finds none of the personal skills and rules under the real home
#   (~/.agents/skills, ~/.claude/skills, ~/.claude/CLAUDE.md), which it would
#   otherwise load, and keeps its settings, account login and session logs
#   there. Muse Code's settings for these (context.foreign_personal_*) don't
#   stop them, and ~/.agents/skills counts as its own;
# - turns Muse Code's OS sandbox off: it keeps .git read-only, and a worktree
#   commits into the main checkout's .git;
# - turns tool approvals off too, so the session runs unattended: approval mode
#   MUSE_APPROVAL (default never, in which nothing stops for review; on-request
#   or untrusted bring the prompts back). The git hooks still refuse a wrong
#   commit.
#
# When the session ends, from any checkout:
#   .venv/bin/python tools/export_transcript.py
# It writes into the worktree and prints the command that commits the files.
set -euo pipefail

MAIN="$(cd "$(dirname "$0")/../.." && pwd)"
WT="${MUSE_WORKTREE:-$(dirname "$MAIN")/Hackathon-muse}"
MODEL="${MUSE_MODEL:-muse-wonder-1.0}"
EFFORT="${MUSE_EFFORT:-max}"
APPROVAL="${MUSE_APPROVAL:-never}"
CONTRIBUTOR="${MUSE_CONTRIBUTOR:-on}"
AUTH="${MUSE_AUTH:-key}"
MUSE_HOME="${MUSE_HOME:-$HOME/.muse-spark}"  # also read by tools/provenance.py
TAG="muse-code-era"
ENV_FILE="$MAIN/legal-agent-workshop/agent/.env"  # also scanned by tools/provenance.py
PY="$MAIN/.venv/bin/python"
SUB=""
case "${1:-}" in
  exec) SUB=exec; shift ;;
  login) SUB=login; shift ;;
esac

command -v muse >/dev/null \
  || { echo "Muse Code isn't installed (https://dev.meta.ai/docs/muse-code)" >&2; exit 1; }

# The name the commit subjects start with, and the co-author line (tools/provenance.py).
AUTHOR="$(PYTHONPATH="$MAIN/tools" "$PY" -c 'import sys, provenance as pv; print("\t".join(pv.muse_author(sys.argv[1])))' "$MODEL")"
NAME="${AUTHOR%%$'\t'*}"
COAUTHOR="${AUTHOR#*$'\t'}"
PROMPT="${MUSE_PROMPT:-Run the muse-round skill: read .agents/skills/muse-round/SKILL.md and follow it. You are $NAME ($MODEL). Start every commit subject with \"$NAME:\" and end every commit message with this line: $COAUTHOR}"

unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL
SIGNIN=()
case "$AUTH" in
  account)
    unset META_API_KEY
    SIGNIN=(--contributor-mode "$CONTRIBUTOR") ;;
  key)
    KEY="$("$PY" -c 'import sys; from dotenv import dotenv_values; sys.stdout.write(dotenv_values(sys.argv[1]).get("MODEL_API_KEY") or "")' "$ENV_FILE")"
    [ -n "$KEY" ] || { echo "MODEL_API_KEY is not set in $ENV_FILE" >&2; exit 1; }
    export META_API_KEY="$KEY"
    unset KEY ;;
  *)
    echo "MUSE_AUTH must be account or key, not $AUTH" >&2; exit 1 ;;
esac

isolate_home() {
  mkdir -p "$MUSE_HOME"
  export HOME="$MUSE_HOME"
  # Muse Code would look for its config, data and other agents' folders here first.
  unset XDG_CONFIG_HOME XDG_DATA_HOME XDG_STATE_HOME XDG_CACHE_HOME CLAUDE_CONFIG_DIR CODEX_HOME
}

if [ "$SUB" = login ]; then
  [ "$AUTH" = account ] || { echo "login is for MUSE_AUTH=account" >&2; exit 1; }
  echo "Signing in to Muse Code with home $MUSE_HOME"
  isolate_home
  exec muse login
fi

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
fi
# The submodules the round reads: the model code and the engine's decode path.
for SM in third_party/mlx-lm third_party/vllm-metal; do
  git -C "$WT" submodule update --init --reference "$MAIN/$SM" "$SM"
done
ln -sfn "$MAIN/.venv" "$WT/.venv"
ln -sfn "$MAIN/models" "$WT/models"
[ "$(git -C "$WT" branch --show-current)" = "muse-agent" ] \
  || { echo "$WT is not on branch muse-agent" >&2; exit 1; }

if git merge-base --is-ancestor muse-agent main; then
  git -C "$WT" merge --ff-only --quiet main
elif ! git merge-base --is-ancestor main muse-agent; then
  echo "note: main has commits muse-agent lacks; merge muse-agent into main first so the session sees them" >&2
fi

cd "$WT"
echo "Muse agent session in Muse Code: $NAME, model $MODEL, effort $EFFORT, approvals $APPROVAL, sign-in $AUTH${SIGNIN[*]:+, contributor mode $CONTRIBUTOR}"
echo "Worktree $WT, branch muse-agent at $(git rev-parse --short HEAD), home $MUSE_HOME"
echo "First prompt: $PROMPT"
isolate_home
exec muse ${SUB:+"$SUB"} --model "$MODEL" --reasoning-effort "$EFFORT" ${SIGNIN[@]+"${SIGNIN[@]}"} \
  --workspace "$WT" --trust-workspace --disable-sandbox --approval-mode "$APPROVAL" \
  -c agents.execution_capacity=12 "$@" "$PROMPT"
