# Checkpoints and logs — 01_direct_rl_coinflip

Pointers only — large checkpoint/log artifacts are not committed to this
repo (see `.gitignore`: `checkpoints/`, `logs/*` except `.gitkeep`).

## Drive checkpoint lineage
```text
/content/drive/MyDrive/AISI/checkpoints/
  grpo-step500-structure-consistency-continuation-v3/checkpoint-100
  full-snapshots/step-500/
```
Logical step 500 is the last verified-good state of this lineage — see
`../README.md` for the full result. This checkpoint is later reused as a
fixed diagnostic subject in `../../04_stability_investigation/`.

Full snapshots (`full-snapshots/step-XXX/`) each contain adapter/model
state, optimizer and scheduler state, `trainer_state.json`, exact reward
source, a complete configuration snapshot, raw monitoring history, and a
versioned manifest. Only resume from a snapshot whose `manifest.json`
contains `"roundtrip_verified": true`.

## Narrative log
`logs/development_log.md` (repo root) is the canonical chronological
record. This folder's `README.md` is a summary derived from it, not a
replacement.
