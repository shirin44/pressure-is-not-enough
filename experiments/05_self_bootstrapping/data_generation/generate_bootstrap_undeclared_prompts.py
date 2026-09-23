"""Generate the undeclared-mapping prompt pool for Option 1 self-bootstrapping.

Pure CPU, no GPU/Colab needed. Produces the PROMPTS ONLY (no model
completions) for the large-scale undeclared-mapping generation pass that
notebooks/option1_bootstrap_generation.ipynb will run against checkpoint
130. Reuses two already-verified functions unchanged: multidomain_seed's
declared-example construction (_render/nonce machinery, via a thin
re-implementation limited to fan/valve at an extended length range -- see
below) and undeclared_generalization.remove_declared_mapping, which strips a
declared mapping from an existing MultiDomainExample. No new prompt template
is written; the undeclared prompt text is byte-for-byte what
remove_declared_mapping already produces.

## Domain choice: fan + valve, not lamp

The original Stage 3.5 undeclared test (verified_nonliteral_encoding_rate =
10%) ran on lamp specifically, the domain multidomain_stage3_plan.json holds
out for zero-shot generalization evaluation. Bootstrapping training data
from lamp itself would mean training and evaluating on the same domain --
exactly the contamination this project's methodology has avoided throughout
(see experiments/multidomain_stage3_plan.json's held-out design). Fan and
valve are the domains checkpoint 130 was actually SFT-trained on (with
declared mappings only -- it has never seen an undeclared-mapping example in
training, on any domain), so sampling undeclared attempts on fan/valve keeps
lamp free for continued held-out evaluation of whatever comes out of Stage
4/self-bootstrapping. Whether fan/valve's undeclared success rate actually
matches lamp's 10% is not assumed -- it is checked directly against the
observed rate reported by the generation notebook.

## Volume: why 8,160, not exactly 8,000-10,000

Target: ~800 genuine successes to train on. Two ways to size N from the
lamp baseline (10/100):
  - Point estimate (10%): N = 800 / 0.10 = 8,000.
  - Conservative planning figure, Wilson 95% lower bound on 10/100 =
    5.52%: N = 800 / 0.0552 = ~14,485.
These bracket a wide range because n=100 is a small sample. The combinatorial
ceiling of unique (initial_state, operations) signatures per domain for flip
lengths 3-L is 2*sum(2^length for length in 3..L) -- extending only to L=8
(the original training/eval range) caps out at 1,008/domain (2,016 total
fan+valve), far short of even the point-estimate target. Extending to L=10
caps out at 4,080/domain (8,160 total), landing almost exactly on the
point-estimate figure and squarely inside the requested 8,000-10,000 range,
without stretching sequence length dramatically past the L=3-8 range the
model was actually trained/evaluated on (L=9-10 is a modest, not extreme,
extension). This is a first-pass volume, not a hard ceiling: the generation
notebook reports the observed success rate explicitly, and if it comes in
well under 10%, a second batch extending to L=11-12 (16,352 combined) is a
straightforward follow-up with the same code, just a wider LENGTH_RANGE.

Repeated sampling of the same prompt was not used to inflate volume: the
original Stage 3.5 methodology used greedy decoding (do_sample=False,
confirmed by re-reading notebooks/stage35_third_domain_zero_shot.ipynb's
generation cell), which is deterministic -- resampling an identical prompt
would yield an identical completion, contributing zero new information.
Every attempt in this pool is therefore a distinct (domain, initial_state,
operations) signature.
"""
from __future__ import annotations

import itertools
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
sys.path.insert(0, str(ROOT))

from src.data.multidomain_seed import (  # noqa: E402
    DOMAIN_SPECS,
    MultiDomainExample,
    _nonce_stream,
    _render,
    simulate,
)
from src.data.undeclared_generalization import remove_declared_mapping  # noqa: E402

DEFAULT_SEED = 20260819  # fresh seed, distinct from every training/eval seed used elsewhere in this repo
LENGTH_RANGE = (3, 10)  # extended from the original 3-8; see docstring for the combinatorial justification
DOMAINS = ("fan", "valve")  # lamp excluded deliberately -- reserved for held-out evaluation, see docstring
OUTPUT_DIR = ROOT / "data" / "processed" / "bootstrap_undeclared_fan_valve_v1"


def _all_signatures(domain: str, min_len: int, max_len: int) -> list[tuple[str, tuple[str, ...]]]:
    """Every distinct (initial_state, operations) pair for flip lengths min_len..max_len."""
    states = DOMAIN_SPECS[domain]["states"]
    signatures = []
    for length in range(min_len, max_len + 1):
        for initial in states:
            for operations in itertools.product(("same", "different"), repeat=length):
                signatures.append((initial, operations))
    return signatures


def generate_bootstrap_pool(seed: int = DEFAULT_SEED) -> list:
    """Build the full undeclared prompt pool: every unique fan/valve signature in LENGTH_RANGE.

    Each signature is first rendered as a normal declared MultiDomainExample (reusing _render/
    _nonce_stream exactly as multidomain_seed.generate_multidomain_dataset does, so the underlying
    prompt/target machinery is identical, not reimplemented), then immediately stripped via
    remove_declared_mapping -- the same function used for the original Stage 3.5 lamp test. The nonce
    token pair assigned during declared rendering is discarded by remove_declared_mapping and never
    seen by the model; it exists only because MultiDomainExample requires one.
    """
    rng = random.Random(seed)
    nonces = _nonce_stream(rng)
    pool = []
    for domain in DOMAINS:
        signatures = _all_signatures(domain, *LENGTH_RANGE)
        rng.shuffle(signatures)
        states = DOMAIN_SPECS[domain]["states"]
        for index, (initial, operations) in enumerate(signatures):
            first, second = next(nonces), next(nonces)
            if index % 2:
                first, second = second, first
            mapping = {states[0]: first, states[1]: second}
            expected = tuple(simulate(domain, initial, operations))
            prompt, demo = _render(domain, initial, operations, expected, mapping, corrected_fan_wording=(domain == "fan"))
            declared_row = MultiDomainExample(
                f"bootstrap-{domain}-{index:05d}", "bootstrap", domain, initial, operations,
                expected, first, second, prompt, demo, expected[-1])
            pool.append(remove_declared_mapping(declared_row))
    rng.shuffle(pool)
    return pool


def audit_pool(pool: list) -> dict:
    """Verify uniqueness, domain balance, and absence of declared-mapping leakage."""
    prompts = [row.prompt for row in pool]
    by_domain = {}
    for row in pool:
        by_domain[row.domain] = by_domain.get(row.domain, 0) + 1
    leakage = [row.example_id for row in pool if "Represent " in row.prompt and " using the code " in row.prompt]
    length_hist = {}
    for row in pool:
        n = len(row.operations)
        length_hist[n] = length_hist.get(n, 0) + 1
    return {
        "total": len(pool),
        "unique_prompts": len(set(prompts)),
        "unique_example_ids": len(set(row.example_id for row in pool)),
        "by_domain": by_domain,
        "declared_mapping_leakage": leakage,
        "length_histogram": dict(sorted(length_hist.items())),
        "accepted": (
            len(prompts) == len(set(prompts))
            and len(pool) == len(set(row.example_id for row in pool))
            and not leakage
        ),
    }


def main() -> None:
    pool = generate_bootstrap_pool()
    report = audit_pool(pool)
    print(json.dumps(report, indent=2))
    if not report["accepted"]:
        raise RuntimeError(f"Bootstrap prompt pool failed its own audit: {report}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "example_id": row.example_id,
            "domain": row.domain,
            "initial_state": row.initial_state,
            "operations": list(row.operations),
            "expected_states": list(row.expected_states),
            "final_answer": row.final_answer,
            "prompt": row.prompt,
        }
        for row in pool
    ]
    out_path = OUTPUT_DIR / "prompts.json"
    out_path.write_text(json.dumps({"seed": DEFAULT_SEED, "length_range": list(LENGTH_RANGE),
                                     "domains": list(DOMAINS), "audit": report, "rows": rows}, indent=2))
    print(f"\nWrote {len(rows)} prompts to {out_path}")


if __name__ == "__main__":
    main()
