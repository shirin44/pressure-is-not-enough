"""CPU/GPU token-pool audit for synthetic-trajectory code tokens (Experiment 3, Step 4).

Produces the candidate table required before any synthetic trajectory is generated:
displayed token, tokenizer IDs, token count, whitespace sensitivity, base-policy and
(once Step 0 exists) task-foundation-policy log-probability at representative state
slots, visual/spelling similarity to banned words, semantic/mnemonic relationship,
occurrence in prompt vocabulary, punctuation/alphanumeric category, leakage-classifier
result, and final eligibility decision.

Run on the EC2 instance (GPU available, Qwen2.5-3B-Instruct already cached, pinned
transformers==5.13.1 matches every other script in this project -- avoids any tokenizer
version mismatch against the local .venv's newer transformers).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# ===== Candidate pool =====
# Short, pronounceable, arbitrary-sounding words with no relation to Coin Flip's task
# vocabulary, no relation to any OTHER domain name used elsewhere in this project
# (fan, valve, lamp -- Stage 02/03), and no obvious binary/directional/semantic pairing
# baked in by word choice (paired up only after the individual audit, not by construction).
CANDIDATES = [
    "Lumen", "Rill", "Vex", "Nib", "Glim", "Dorf", "Kip", "Zeb", "Ooma", "Plor",
    "Snick", "Fenn", "Tark", "Wisp", "Crick", "Nomo", "Yelt", "Brix", "Quon", "Zeph",
    "Murn", "Sable", "Trin", "Ovid", "Wexa", "Loft", "Drez", "Ambo", "Yark", "Iven",
]

# ===== Banned-word / leakage checks =====
BANNED_WORDS = [
    "heads", "tails", "head", "tail", "h", "t",
    "coin", "flip", "flips", "same", "different", "state", "side",
]
# The project's whole other multi-domain lineage (02/03) -- must not accidentally reuse
# a domain name from elsewhere in this project as a Coin Flip code token.
OTHER_PROJECT_DOMAIN_WORDS = ["fan", "valve", "lamp", "running", "stopped", "open", "closed", "lit", "dark"]

PROMPT_VOCABULARY_WORDS = [
    "starting", "instructions", "reason", "step", "remains", "unchanged", "changes",
    "flips", "answer", "capitalized", "alphabetic", "previous", "same", "different",
]


def is_mnemonic_adjacent(code: str, real_word: str) -> bool:
    """Same rule as the project's established leakage classifier (05/06/07): shares
    first letter, or one is a prefix of the other."""
    code_cf, word_cf = code.casefold(), real_word.casefold()
    if not code_cf or not word_cf:
        return False
    if code_cf[0] == word_cf[0]:
        return True
    if word_cf.startswith(code_cf) or code_cf.startswith(word_cf):
        return True
    return False


def edit_distance(a: str, b: str) -> int:
    a, b = a.casefold(), b.casefold()
    if len(a) < len(b):
        return edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        current_row = [i + 1]
        for j, cb in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def leakage_check(token: str) -> dict:
    all_banned = BANNED_WORDS + OTHER_PROJECT_DOMAIN_WORDS
    mnemonic_hits = [w for w in all_banned if is_mnemonic_adjacent(token, w)]
    min_edit = min((edit_distance(token, w) for w in BANNED_WORDS), default=99)
    closest_banned = min(BANNED_WORDS, key=lambda w: edit_distance(token, w))
    in_prompt_vocab = token.casefold() in PROMPT_VOCABULARY_WORDS
    is_alpha_only = token.isalpha()
    return {
        "mnemonic_adjacent_hits": mnemonic_hits,
        "min_edit_distance_to_banned_word": min_edit,
        "closest_banned_word": closest_banned,
        "in_prompt_vocabulary": in_prompt_vocab,
        "alpha_only": is_alpha_only,
        "passes_leakage_classifier": len(mnemonic_hits) == 0 and min_edit >= 3 and not in_prompt_vocab and is_alpha_only,
    }


def tokenizer_audit(tokenizer, token: str) -> dict:
    no_space_ids = tokenizer.encode(token, add_special_tokens=False)
    space_ids = tokenizer.encode(" " + token, add_special_tokens=False)
    return {
        "token_ids_no_leading_space": no_space_ids,
        "n_tokens_no_leading_space": len(no_space_ids),
        "token_ids_with_leading_space": space_ids,
        "n_tokens_with_leading_space": len(space_ids),
        "whitespace_changes_tokenization": no_space_ids != space_ids,
        "is_single_token_either_way": len(no_space_ids) == 1 or len(space_ids) == 1,
    }


def logprob_at_state_slot(model, tokenizer, token: str, device) -> float | None:
    """log P(token | 'Step 1: The state flips. State: ') under the given model --
    a representative state-slot context matching the established prompt convention."""
    import torch
    context = "Step 1: The state flips. State:"
    context_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
    target_ids = tokenizer.encode(" " + token, add_special_tokens=False)
    if not target_ids:
        return None
    with torch.inference_mode():
        out = model(context_ids)
        logits = out.logits[0, -1]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        return float(log_probs[target_ids[0]].item())


def build_table(tokenizer, model=None, device=None) -> list[dict]:
    rows = []
    for token in CANDIDATES:
        row = {"token": token}
        row.update(tokenizer_audit(tokenizer, token))
        row.update(leakage_check(token))
        if model is not None:
            row["base_policy_logprob_at_state_slot"] = logprob_at_state_slot(model, tokenizer, token, device)
        else:
            row["base_policy_logprob_at_state_slot"] = None
        # NOTE: "whitespace_changes_tokenization" is reported (per the requested table
        # columns) but NOT used as a disqualifier -- BPE tokenizers virtually always
        # tokenize a capitalized word differently at string-start vs. after a space (this
        # is normal leading-space-token behavior, not a defect). What actually matters is
        # the WITH-LEADING-SPACE token count, since every real usage is "State: <token>"
        # (always preceded by a space) -- checked as n_tokens_with_leading_space below.
        eligible = (
            row["passes_leakage_classifier"]
            and row["n_tokens_with_leading_space"] <= 2
        )
        row["eligible"] = eligible
        row["eligibility_reason"] = (
            "passes all checks" if eligible else
            ("fails leakage classifier" if not row["passes_leakage_classifier"] else
             "too many tokens in actual usage context (after a leading space)")
        )
        rows.append(row)
    return rows


if __name__ == "__main__":
    from transformers import AutoTokenizer

    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)

    model = None
    device = None
    try:
        import torch
        from transformers import AutoModelForCausalLM
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, dtype=torch.bfloat16, device_map={"": 0},
                low_cpu_mem_usage=True, use_safetensors=True, trust_remote_code=False)
            model.eval()
            print({"model_loaded_for_logprobs": True})
        else:
            print({"model_loaded_for_logprobs": False, "reason": "no CUDA GPU -- tokenizer-only pass"})
    except Exception as e:
        print({"model_loaded_for_logprobs": False, "reason": str(e)})

    rows = build_table(tokenizer, model, device)
    eligible = [r for r in rows if r["eligible"]]
    print(f"\n{'token':10s} {'n_tok':6s} {'ws_sens':8s} {'leak_ok':8s} {'min_ed':7s} {'logprob':10s} {'eligible':9s}")
    for r in rows:
        lp = r["base_policy_logprob_at_state_slot"]
        lp_str = f"{lp:.2f}" if lp is not None else "n/a"
        print(f"{r['token']:10s} {r['n_tokens_with_leading_space']:<6d} "
              f"{str(r['whitespace_changes_tokenization']):8s} {str(r['passes_leakage_classifier']):8s} "
              f"{r['min_edit_distance_to_banned_word']:<7d} {lp_str:10s} {str(r['eligible']):9s}")
    print(f"\n{len(eligible)}/{len(rows)} candidates eligible.")

    out_path = Path.home() / "aisi_checkpoints" / "exp3_token_pool_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print("Saved:", out_path)
