# Exploration: round 1

**Version:** v4 · 2026-11-03T21:30:00Z · skill version 3

## Tracker

| Territory | Agent | Status | Inefficiencies | Updated |
|---|---|---|---|---|
| `model` | `explorer-model` | done | `ineff-model-1` (superseded by ineff-runs-2), `ineff-model-2` (confirmed) | 2026-11-03 21:30Z |
| `switch-layers` | `explorer-switch-layers` | done | `ineff-switch-layers-1` (confirmed), `-2` (confirmed) | 2026-11-03 21:30Z |
| `quantized-kernels` | `explorer-quantized-kernels` | done | `ineff-quantized-kernels-1` (confirmed), `-2` (superseded by ineff-switch-layers-2) | 2026-11-03 21:30Z |
| `other-kernels` | `explorer-other-kernels` | done | — (kernel notes on runs-2, model-2, runs-3) | 2026-11-03 21:30Z |
| `vllm-metal` | `explorer-vllm-metal` | done | `ineff-vllm-metal-1` (superseded by ineff-findings-1), `-2` (superseded by ineff-runs-4), `-3` (confirmed), `-4` (confirmed) | 2026-11-03 21:30Z |
| `grader` | `explorer-grader` | done | — (grader notes on runs-1/2/3, model-2) | 2026-11-03 21:30Z |
| `findings` | `explorer-findings` | done | `ineff-findings-1` (superseded by ineff-runs-2 and ineff-model-2), `-2` (confirmed), `-3` (confirmed) | 2026-11-03 21:30Z |
| `runs` | `explorer-runs` | done | `ineff-runs-1` (confirmed), `-2` (confirmed), `-3` (refuted), `-4` (confirmed) | 2026-11-03 21:30Z |

Verdicts: 11 confirmed, 1 refuted (`ineff-runs-3`: micro gaps measure kernel choice, not pair order), 5 superseded, 0 open.

## Versions

- v4 · 2026-11-03T21:30:00Z · verifier done: 11 confirmed, runs-3 refuted · changed: ineff-runs-1 v3, ineff-runs-2 v3, ineff-runs-3 v3, ineff-runs-4 v3, ineff-model-2 v3, ineff-findings-2 v4, ineff-findings-3 v3, ineff-switch-layers-1 v2, ineff-switch-layers-2 v2, ineff-quantized-kernels-1 v3, ineff-vllm-metal-3 v3, ineff-vllm-metal-4 v3
- v3 · 2026-11-03T21:27:27Z · all explorers done; findings-1 superseded by runs-2+model-2; 8 premature confirmations sent to verifier · changed: territory-switch-layers v2, territory-quantized-kernels v3, territory-other-kernels v2, territory-vllm-metal v2, territory-grader v2, territory-findings v4, ineff-findings-1 v3, ineff-findings-2 v3, ineff-quantized-kernels-1 v2
- v2 · 2026-11-03T21:21:13Z · model and runs done; 8 inefficiencies posted · changed: territory-model v2, territory-runs v2, ineff-model-1 v2, ineff-model-2 v1, ineff-runs-1 v1, ineff-runs-2 v1, ineff-runs-3 v1, ineff-runs-4 v1, territory-vllm-metal v1, ineff-vllm-metal-1 v1, ineff-vllm-metal-2 v1
- v1 · 2026-11-03T21:17:35Z · opened exploration, 8 territories started · changed: none
