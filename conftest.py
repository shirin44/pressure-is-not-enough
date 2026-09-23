"""Repo-root pytest config: excludes files that match pytest's default discovery
globs (`test_*.py` / `*_test.py`) but are not pytest test suites at all -- standalone
scripts meant to be run directly, not imported/collected. Importing them under pytest
either crashes collection (live AWS/subprocess calls with a module-level `sys.exit`)
or raises at import time (a hard `if not torch.cuda.is_available(): raise ...` guard,
correct for their intended standalone GPU use, fatal in a CPU-only CI runner).

Each entry's own docstring/module already says how it's actually meant to be run;
this file changes nothing about that -- it only stops pytest's automatic, unintended
discovery of them."""

collect_ignore = [
    # Live AWS/subprocess integration test -- see its own docstring
    # ("Run: python3 scripts/test_gpu_teardown.py"). Makes real AWS calls and calls
    # sys.exit(1) at module level on failure, which is fine run directly but crashes
    # pytest's collector (an uncaught SystemExit during import) when auto-discovered.
    'scripts/test_gpu_teardown.py',
    # GPU-only standalone scripts (hard `torch.cuda.is_available()` guard at module
    # level, by design, for their real intended use) that happen to match pytest's
    # `*_test.py` discovery glob, not just its own experiment directory's README-
    # documented standalone invocation.
    'experiments/07_positive_signal_annealed_reward/aws_recovered/stage08_entropy_test.py',
    'experiments/09e_same_different_llama/llama_causal_flip_test.py',
    'experiments/11_concealment_tests/semantic_monitor_v2/paraphrase_resumption_test.py',
]
