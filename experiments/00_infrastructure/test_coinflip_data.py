from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.coinflip import generate_coinflip_example, generate_dataset


def main() -> None:
    print("Generating 10 coin-flip examples...\n")
    examples = generate_dataset(10, (1, 6))
    for idx, (prompt, answer) in enumerate(examples, start=1):
        print(f"Example {idx}")
        print("-" * 60)
        print(prompt)
        print(f"Ground truth answer: {answer}")
        print()

    print("Manual verification check")
    print("-" * 60)
    prompt, answer = generate_coinflip_example(3, seed=7)
    print(prompt)
    print(f"Ground truth answer: {answer}")


if __name__ == "__main__":
    main()
