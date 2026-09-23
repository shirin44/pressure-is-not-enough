from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.latch_seed import generate_latch_seed_dataset, write_latch_seed_dataset


def main() -> None:
    output_dir = REPO_ROOT / "data" / "processed" / "latch_seed_v5_explicit_transitions"
    report = write_latch_seed_dataset(output_dir)
    train, heldout = generate_latch_seed_dataset()
    print("===== LATCH SEED DATASET ACCEPTANCE GATE =====")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\n===== REPRESENTATIVE TRAINING EXAMPLES =====")
    for example in (train[0], train[1], train[2], heldout[0], heldout[1]):
        print(f"\n--- {example.example_id} ---")
        print(example.prompt)
        print("\nWORKED DEMONSTRATION")
        print(example.demonstration)
        print(
            "MAPPING:",
            {"Locked": example.token_for_locked, "Unlocked": example.token_for_unlocked},
        )
    print(f"\nACCEPTED. Files written to {output_dir}")


if __name__ == "__main__":
    main()
