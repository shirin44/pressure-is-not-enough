# Archive

Not part of the main experimental narrative — kept for reference, not for
a future reader to work through in order.

- `experiments/archive/notebooks/`, `experiments/archive/scripts/` —
  Drive housekeeping utilities (checkpoint inventory, approved cleanup
  runs). Not experimental results; don't fit any numbered stage.
- `experiments/archive/scripts/run_experiment.sh` — dead scaffold from the
  initial repo commit. Referenced a `src.eval.run` pipeline that was never
  implemented (`src/eval/` has only ever contained an empty `__init__.py`).
  Confirmed via repo-wide grep before archiving: no other file imports or
  invokes it.
- `experiments/archive/misc/default.yaml` — the config file that scaffold
  pointed to, same status.

For the actual chronological experiment history, start at
`../00_infrastructure/README.md` and follow the numbered folders, or read
the top-level `README.md`'s table of contents.
