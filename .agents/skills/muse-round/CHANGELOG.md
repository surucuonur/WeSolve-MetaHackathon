# Versions of the muse-round skill

*Written by Claude Code (Claude Opus 5.5), directed by surucuonur. [Who made what](../../../PROVENANCE.md).*

The skill is this folder: `SKILL.md` for the lead, the role cards and `tracking.md`, versioned together. Claude raises the version with every change, and commits it on main before the round that first uses it. Each round records the version it ran in its READMEs and in `muse-spark/summary.md`. Newest first:

| Version | Date | Commit | What changed |
|---|---|---|---|
| 3 | 2026-11-03 | "Version Muse Spark's shared documents and its round skill" | Every shared document carries a version. Each checkpoint commit is one phase version, named in its subject, with a changelog line in the README. Entries carry their own versions. The skill carries this number ([ADR 0007](../../../claude/adr/0007-muse-spark-swarm.md)). |
| 2 | 2026-11-03 | `d657a1f`, fixed in `8620ab4` | The swarm: explorers, a verifier, competing builders and boundary agents. Phase folders with trackers, entries and logs; role cards and `tracking.md` ([ADR 0007](../../../claude/adr/0007-muse-spark-swarm.md)). `8620ab4`: commit only the phase folders that exist. |
| 1 | 2026-11-02 | `5540b8c` | A lead with short-lived helpers: four lenses, a verifier, two drafters and a reviewer per slot; `diagnosis.md` and `strategy.md` ([ADR 0006](../../../claude/adr/0006-muse-spark-in-muse-code.md)). surucuonur's 4-minute trial session `01a1a2b8-03a0-7890-baf7-9febe7b8d245` ran it on 2026-11-03 and committed nothing. |
