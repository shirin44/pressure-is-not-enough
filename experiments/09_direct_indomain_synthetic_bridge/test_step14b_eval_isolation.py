"""Regression test for the pre-flight confirmation given before launching Step 14b's
full run: injected rows must never be reachable from the milestone-evaluation code
path. Static, AST-based, and applied to every actual training script in this
directory (dry run and full run alike) -- not a one-time manual trace that could
silently stop being true after a later edit.

Why this matters (per the explicit pre-launch requirement): if injection-related
functions were ever called from inside evaluate_milestone/_evaluate_cohort/
run_eval_generation, a milestone could show "non-literal completions present"
purely because the injected, already-verified trajectory was being counted --
a false and misleading signal about whether the MODEL learned anything, in the
same family as this project's prior eval/attribution mistakes (the Step 13 stale
eval-count bug, the eval/bank overlap)."""
import ast
from pathlib import Path

INJECTION_IDENTIFIERS = {
    "inject_into_generation_output",
    "_injection_aware_generate",
    "select_injection_indices",
    "build_injected_completion_ids",
}
EVAL_FUNCTION_NAMES = {"run_eval_generation", "_evaluate_cohort", "evaluate_milestone"}

HERE = Path(__file__).resolve().parent
INJECTION_SCRIPTS = sorted(HERE.glob("step14b_bridge_reward_gate_injection_*.py"))


def _names_referenced_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
    }


def test_at_least_one_injection_training_script_exists_to_check():
    assert INJECTION_SCRIPTS, "no step14b_bridge_reward_gate_injection_*.py found to verify"


def test_evaluation_functions_never_reference_injection_machinery():
    checked_any_eval_function = False
    for script_path in INJECTION_SCRIPTS:
        tree = ast.parse(script_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in EVAL_FUNCTION_NAMES:
                checked_any_eval_function = True
                referenced = _names_referenced_in(node)
                leaked = referenced & INJECTION_IDENTIFIERS
                assert not leaked, (
                    f"{script_path.name}::{node.name} references injection machinery "
                    f"{leaked} -- injected rows could leak into evaluation/taxonomy counts"
                )
    assert checked_any_eval_function, "no evaluate_milestone/_evaluate_cohort/run_eval_generation found to check"


def test_evaluation_generation_calls_model_generate_directly_not_the_trainer():
    """Confirms run_eval_generation's generation call is `model.generate(...)`, the
    same object patched only with the stopping-criteria wrapper -- never
    `diagnostic_trainer._generate(...)`, which is what injection is installed on."""
    for script_path in INJECTION_SCRIPTS:
        tree = ast.parse(script_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_eval_generation":
                calls = [
                    ast.dump(n.func) for n in ast.walk(node)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "generate"
                ]
                assert calls, f"{script_path.name}: run_eval_generation has no .generate(...) call to check"
                for call_repr in calls:
                    assert "diagnostic_trainer" not in call_repr, (
                        f"{script_path.name}: run_eval_generation calls the TRAINER's generate, "
                        f"not the model's -- this would route through injection"
                    )
