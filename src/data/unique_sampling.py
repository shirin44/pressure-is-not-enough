"""Standing guard against the "draw-random-and-dedupe until N unique" infinite-loop bug.

The bug: a generator loop draws a random scenario, checks it against a `seen` set, and
retries on a collision, looping `while len(pool) < n_requested`. If the underlying
combinatorial space of distinct possible scenarios is smaller than `n_requested`, this
loop never terminates (or, just below the threshold, takes exponentially many draws to
find the last few unique items -- the coupon collector's problem).

This has hit this project twice, independently, both caught only via a `py-spy` stack
dump after minutes of silent hanging: the Coin Flip bootstrap-generation prompt pool
(experiments/05_self_bootstrapping, requesting 6,000 unique prompts from a 248-prompt
space) and Experiment 1's flip-length sweep (requesting 40 unique scenarios at flip-length
2, where only 8 exist). Both were fixed the same way: compute the space size first,
either cap the request or enumerate it exhaustively instead of probing.

Use these functions BEFORE any draw-and-dedupe loop, not after it hangs.
"""
from __future__ import annotations

import random
from typing import Sequence, TypeVar

T = TypeVar("T")


def combinatorial_space_size(*dimension_sizes: int) -> int:
    """Multiply out independent choice-dimension sizes into a total scenario count.

    Example: Coin Flip's (starting_state, operations) space at a given flip length is
    combinatorial_space_size(2, 2 ** n_flips) -- 2 starting states, 2**n_flips operation
    sequences.
    """
    if not dimension_sizes:
        raise ValueError("at least one dimension size is required")
    size = 1
    for d in dimension_sizes:
        if d < 0:
            raise ValueError(f"dimension sizes must be non-negative, got {d}")
        size *= d
    return size


def check_unique_sample_feasible(n_requested: int, space_size: int, *, context: str = "") -> None:
    """Raise immediately if `n_requested` unique items cannot possibly be drawn from a
    space of size `space_size`. Call this as the FIRST line of any function that is about
    to draw-and-dedupe random scenarios -- it turns an infinite hang into an instant,
    readable error at the call site.
    """
    if n_requested > space_size:
        label = f" ({context})" if context else ""
        raise ValueError(
            f"Requested {n_requested} unique scenarios{label}, but the underlying "
            f"combinatorial space only contains {space_size}. A draw-and-dedupe loop "
            f"would never terminate (or would take exponentially many draws to find the "
            f"last few). Either reduce the request to <= {space_size}, or use "
            f"cap_unique_sample_size() to size the request automatically, or use "
            f"sample_unique() to draw from an exhaustively enumerated space instead of "
            f"probing for uniqueness."
        )


def cap_unique_sample_size(n_requested: int, space_size: int) -> int:
    """Return min(n_requested, space_size) -- the explicit, silent-hang-free alternative
    to check_unique_sample_feasible() for callers who want a smaller-than-requested pool
    rather than a raised error when the space is too small."""
    return min(n_requested, space_size)


def sample_unique(scenarios: Sequence[T], n_requested: int, seed: int) -> list[T]:
    """Safely sample up to n_requested unique items from an ALREADY exhaustively
    enumerated `scenarios` sequence: shuffle, then slice. This is the safe replacement
    for a draw-random-and-check-`seen`-set loop -- it can never hang, because it never
    probes for uniqueness; it enumerates once and takes what's there. Returns
    min(n_requested, len(scenarios)) items -- always check the returned length against
    what was requested if the caller needs to know whether it was capped.
    """
    rng = random.Random(seed)
    shuffled = list(scenarios)
    rng.shuffle(shuffled)
    return shuffled[:n_requested]
