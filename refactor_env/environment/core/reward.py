"""
reward.py  —  Pure reward computation for refactor-env.

This module has ZERO I/O.  It takes:
  - GraderResults  (from metrics.py)
  - RefactorState  (from models.py — for noop count, accumulated penalty)
  - ScenarioSpec   (from registry — for weights, penalty magnitudes)

And returns a RefactorReward (from models.py).

The formula (from design.md):
  raw = acc_w * acc + qual_w * qual + eff_w * eff + fmt_w * fmt
  penalty = min(sum_of_penalties, PENALTY_CAP)
  score = clamp(raw - penalty, 0.0, 1.0)

Penalty rules (from grading_system.md):
  syntax_error     : -penalty_config.syntax_error  (once per submit)
  test_regression  : -penalty_config.broken_import (if tests were passing before)
  repeated_noop    : -penalty_config.repeated_noop  (per noop above threshold)
  invariant_breach : checked per invariant; capped separately

All penalty values come from ScenarioSpec.penalties — never hardcoded here.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

from refactor_env.models import RefactorReward, RewardBreakdown

if TYPE_CHECKING:
    from refactor_env.environment.core.metrics import GraderResults
    from refactor_env.environment.registry.scenario import ScenarioSpec
    from refactor_env.models import RefactorState

# Caps to prevent penalty from going catastrophically negative
_TOTAL_PENALTY_CAP = 0.70  # reward cannot go below (raw - 0.70)
_NOOP_PENALTY_CAP = 0.30  # max contribution from noop penalties
_INVARIANT_PENALTY_CAP = 0.40  # max contribution from invariant violations


def compute_reward(
    grader_results: "GraderResults",
    state: "RefactorState",
    spec: "ScenarioSpec",
    invariant_violations: List[str],
    is_submit: bool = False,
) -> RefactorReward:
    """
    Compute the final RefactorReward from grader results + state context.

    Parameters
    ----------
    grader_results:
        Output of collect_metrics().
    state:
        Current RefactorState — provides consecutive_noop_count,
        accumulated_penalty, and step history.
    spec:
        ScenarioSpec — provides reward_weights and penalties config.
    invariant_violations:
        List of violation messages from ScenarioSpec.check_invariants().
        Can be empty.
    is_submit:
        True when called from the submit action.  Syntax penalty only applies
        on submit (not on intermediate steps, to allow WIP editing).

    Returns
    -------
    RefactorReward  (frozen dataclass, safe to cache)
    """
    w = spec.reward_weights
    pc = spec.penalties

    # ── 1. Raw weighted score ─────────────────────────────────────────────────
    acc = grader_results.score_for("acc")
    qual = grader_results.score_for("qual")
    eff = grader_results.score_for("eff")
    fmt = grader_results.score_for("fmt")

    raw = (w.acc * acc) + (w.qual * qual) + (w.eff * eff) + (w.fmt * fmt)

    # ── 2. Penalty accumulation ───────────────────────────────────────────────
    bd = RewardBreakdown(
        accuracy=acc,
        quality=qual,
        efficiency=eff,
        format_ok=fmt,
        weighted_raw=round(raw, 4),
    )

    penalty_syntax = 0.0
    penalty_regression = 0.0
    penalty_noop = 0.0
    penalty_invariant = 0.0

    # 2a. Syntax error penalty (submit-only)
    if is_submit and grader_results.has_syntax_error:
        penalty_syntax = pc.syntax_error

    # 2b. Test regression penalty
    # Applied when tests were passing at baseline (acc was 1.0 at episode start)
    # but are now failing. We detect this by checking if baseline had no failures
    # and current has failures.
    if is_submit and _is_test_regression(grader_results, state):
        penalty_regression = pc.broken_import  # reuses broken_import magnitude

    # 2c. Repeated no-op penalty
    # Each noop above threshold (1) incurs a penalty, capped at _NOOP_PENALTY_CAP
    noop_count = max(0, state.consecutive_noop_count - 1)
    penalty_noop = min(noop_count * pc.repeated_noop, _NOOP_PENALTY_CAP)

    # 2d. Invariant breach penalty
    # Each unique violation message incurs a flat penalty, capped
    unique_violations = len(set(invariant_violations))
    penalty_invariant = min(
        unique_violations * pc.syntax_error * 0.67,  # ~0.20 per breach
        _INVARIANT_PENALTY_CAP,
    )

    total_penalty = min(
        penalty_syntax + penalty_regression + penalty_noop + penalty_invariant,
        _TOTAL_PENALTY_CAP,
    )

    # ── 3. Final score ────────────────────────────────────────────────────────
    score = max(0.0, min(1.0, raw - total_penalty))
    # partial_credit = raw before penalty (what the agent earned before deductions)
    partial_credit = max(0.0, min(1.0, raw))

    # ── 4. Fill breakdown ─────────────────────────────────────────────────────
    bd = RewardBreakdown(
        accuracy=round(acc, 4),
        quality=round(qual, 4),
        efficiency=round(eff, 4),
        format_ok=round(fmt, 4),
        weighted_raw=round(raw, 4),
        penalty_syntax_error=round(penalty_syntax, 4),
        penalty_test_regression=round(penalty_regression, 4),
        penalty_noop_loop=round(penalty_noop, 4),
        penalty_invariant_breach=round(penalty_invariant, 4),
        total_penalty=round(total_penalty, 4),
    )

    # ── 5. Build feedback string ──────────────────────────────────────────────
    feedback = _build_feedback(bd, invariant_violations, grader_results)

    return RefactorReward(
        score=round(score, 4),
        partial_credit=round(partial_credit, 4),
        penalty=round(total_penalty, 4),
        breakdown=bd,
        feedback=feedback,
        done=is_submit,
        info={
            "syntax_error_files": grader_results.syntax_error_files,
            "broken_imports": grader_results.broken_imports,
            "invariant_violations": invariant_violations,
            "noop_count": state.consecutive_noop_count,
            "step_count": state.step_count,
            "lint_errors": grader_results.lint_summary.total_errors,
            "tests_passed": grader_results.test_summary.passed,
            "tests_total": grader_results.test_summary.total,
        },
    )


def compute_step_reward(
    grader_results: "GraderResults",
    state: "RefactorState",
    spec: "ScenarioSpec",
) -> RefactorReward:
    """
    Lightweight intermediate reward for non-submit steps.

    Runs the same formula but:
      - syntax penalty is suppressed (agent may be mid-edit)
      - is_submit=False so done=False
      - invariants are not re-checked (checked only on submit)
    """
    return compute_reward(
        grader_results=grader_results,
        state=state,
        spec=spec,
        invariant_violations=[],
        is_submit=False,
    )


# ── Private helpers ───────────────────────────────────────────────────────────


def _is_test_regression(
    grader_results: "GraderResults",
    state: "RefactorState",
) -> bool:
    """
    True if tests are now failing AND baseline_metrics shows they were passing.
    Avoids false positives when tests were already failing at episode start.
    """
    baseline_acc = state.baseline_metrics.get("acc", None)
    if baseline_acc is None:
        return False  # no baseline recorded — can't determine regression
    ts = grader_results.test_summary
    current_failing = (ts.failed + ts.errors) > 0
    was_clean = baseline_acc >= 1.0
    return was_clean and current_failing


def _build_feedback(
    bd: "RewardBreakdown",
    violations: List[str],
    gr: "GraderResults",
) -> str:
    """
    Build a dense one-paragraph feedback string for the observation.
    Designed to be directly usable in an LLM prompt.
    """
    parts: List[str] = []

    parts.append(
        f"Score: {bd.weighted_raw:.2f} raw "
        f"→ {bd.weighted_raw - bd.total_penalty:.2f} after penalties."
    )
    parts.append(
        f"Components: acc={bd.accuracy:.2f} qual={bd.quality:.2f} "
        f"eff={bd.efficiency:.2f} fmt={bd.format_ok:.2f}."
    )

    if gr.lint_summary.total_errors > 0:
        top_codes = sorted(
            gr.lint_summary.error_by_code.items(),
            key=lambda x: -x[1],
        )[:3]
        top_str = ", ".join(f"{c}:{n}" for c, n in top_codes)
        parts.append(
            f"Lint: {gr.lint_summary.total_errors} violations remaining "
            f"(top: {top_str})."
        )
    else:
        parts.append("Lint: 0 violations — all rules clean.")

    ts = gr.test_summary
    if ts.total > 0:
        parts.append(
            f"Tests: {ts.passed}/{ts.total} passed"
            + (f", {ts.failed} failed" if ts.failed else "")
            + (f", {ts.errors} errors" if ts.errors else "")
            + "."
        )

    if bd.total_penalty > 0:
        penalty_parts = []
        if bd.penalty_syntax_error:
            penalty_parts.append(
                f"syntax_error={bd.penalty_syntax_error:.2f}"
                f" (files: {', '.join(gr.syntax_error_files)})"
            )
        if bd.penalty_test_regression:
            penalty_parts.append(f"test_regression={bd.penalty_test_regression:.2f}")
        if bd.penalty_noop_loop:
            penalty_parts.append(f"noop_loop={bd.penalty_noop_loop:.2f}")
        if bd.penalty_invariant_breach:
            penalty_parts.append(f"invariant_breach={bd.penalty_invariant_breach:.2f}")
        parts.append(f"Penalties: {'; '.join(penalty_parts)}.")

    if violations:
        parts.append(f"Invariant failures: {'; '.join(violations[:3])}.")

    return " ".join(parts)
