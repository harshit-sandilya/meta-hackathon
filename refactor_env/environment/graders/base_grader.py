"""
Base grader contract.

Every grader in this package:
  1. Inherits from BaseGrader
  2. Implements compute_metrics() — runs tools, has side effects
  3. Implements grade() — pure function, no side effects
  4. Implements gold_standard() — declares what score=1.0 looks like

The separation between compute_metrics() and grade() is critical:
  - It allows unit tests to inject fake metrics without running real tools
  - It allows the environment to cache metrics independently of scoring
  - grade() must be deterministic: same inputs → same GradeResult always

Scoring contract (enforced by BaseGrader.grade()):
  - All scores are in [0.0, 1.0]
  - score = 0.0 means no improvement from baseline (or regression)
  - score = 1.0 means gold standard reached
  - Negative scores are never returned — regressions are captured
    as score=0.0 and surfaced in raw_metrics for reward.py to penalize
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class GradeResult:
    """
    The output of a single grader for a single step.

    All graders return exactly this — reward.py only ever touches GradeResult,
    never the internals of any grader.
    """

    # Primary score: improvement from baseline, clamped to [0.0, 1.0]
    score: float

    # How far from gold standard. 0.0 = gold reached. 1.0 = no progress at all.
    # Always equals (1.0 - score) but surfaced explicitly for reward shaping.
    gold_distance: float

    # Raw metric snapshots — stored for feedback generation and debugging.
    # Keys are grader-specific (e.g. "violations_now", "violations_baseline").
    raw_baseline: dict[str, Any]
    raw_current: dict[str, Any]

    # Per-metric deltas: current - baseline. Positive = improvement (fewer violations,
    # lower complexity). Negative = regression. Stored as signed values.
    delta: dict[str, Any]

    # Human-readable explanation sent back to the LLM agent in RefactorObservation.
    # Should be one or two sentences describing what changed and what remains.
    feedback: str

    # Whether this grader considers the episode solved from its dimension alone.
    # reward.py uses the AND of all active graders' solved flags to set done=True
    # when combined with the acc (test pass) signal.
    solved: bool = False

    # Whether the current state is a regression from baseline.
    # reward.py uses this to add a penalty on top of score=0.0.
    is_regression: bool = False

    # Optional: grader-specific sub-scores (e.g. for ComplexityGrader: cc_score,
    # time_score, space_score). Not used by reward.py — diagnostic only.
    sub_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Enforce invariants
        self.score = _clamp(self.score)
        self.gold_distance = _clamp(self.gold_distance)
        # Ensure gold_distance is consistent with score
        expected_gd = round(1.0 - self.score, 10)
        if abs(self.gold_distance - expected_gd) > 1e-6:
            self.gold_distance = expected_gd


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class BaseGrader(ABC):
    """
    Abstract contract all graders must implement.

    Lifecycle per episode
    ─────────────────────
    reset()
      └─ grader.compute_metrics(repo_path, config, cache) → baseline_metrics
         grader._baseline = baseline_metrics   (stored by environment)

    step(edit_file action)
      └─ grader.compute_metrics(repo_path, config, cache) → current_metrics
         grader.grade(baseline_metrics, current_metrics, config) → GradeResult

    The grader itself is stateless between calls — all state lives in the
    metrics dicts passed in. This makes graders safe to call in parallel
    and trivial to unit test.
    """

    # Grader identifier — must match the key used in scenario.yaml graders: block
    # and in graders/__init__.py registry. Set as class variable in each subclass.
    grader_id: str = "base"

    # ---------------------------------------------------------------------------
    # Methods every subclass must implement
    # ---------------------------------------------------------------------------

    @abstractmethod
    def compute_metrics(
        self,
        repo_path: object,  # pathlib.Path — typed as object to avoid circular import
        config: dict[str, Any],
        cache: object,  # MetricCache instance
    ) -> dict[str, Any]:
        """
        Run static analysis / tool execution and return raw metrics dict.

        This method IS allowed to have side effects (running subprocesses,
        reading files). It MUST be deterministic given the same filesystem
        state — same repo contents → same metrics dict.

        The returned dict is passed verbatim as `baseline` and `current`
        to grade(). Keys and value types are grader-specific but must be
        JSON-serializable (for state() logging).

        Parameters
        ----------
        repo_path : Path to the sandbox repo root.
        config    : The scenario config dict from scenario.yaml (already parsed).
                    Graders read their own keys from config (e.g.
                    config["graders"]["lint"]["weight"]) plus shared keys like
                    config["public_api"], config["exclude_patterns"].
        cache     : MetricCache instance. Use cache.get_or_compute() for any
                    expensive tool call.
        """

    @abstractmethod
    def grade(
        self,
        baseline: dict[str, Any],
        current: dict[str, Any],
        config: dict[str, Any],
    ) -> GradeResult:
        """
        Compute score from baseline and current metrics. Pure function.

        MUST be deterministic: same (baseline, current, config) → same GradeResult.
        MUST NOT touch the filesystem, run subprocesses, or have side effects.
        MUST return score in [0.0, 1.0] (enforced by GradeResult.__post_init__).

        Parameters
        ----------
        baseline : Metrics dict from compute_metrics() at reset().
        current  : Metrics dict from compute_metrics() at current step.
        config   : Same scenario config dict passed to compute_metrics().
        """

    @abstractmethod
    def gold_standard(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Return a metrics dict representing the gold standard (score = 1.0).

        Used for documentation and for computing gold_distance in tests.
        The returned dict has the same keys as compute_metrics() output.
        """

    # ---------------------------------------------------------------------------
    # Optional hook — subclasses may override
    # ---------------------------------------------------------------------------

    def is_applicable(self, config: dict[str, Any]) -> bool:
        """
        Return True if this grader should run for the given scenario config.

        Default: True if this grader_id appears in config["graders"] with
        weight > 0. Subclasses can add further conditions (e.g. CoverageGrader
        checks that a tests/ directory exists).
        """
        graders_cfg = config.get("graders", {})
        entry = graders_cfg.get(self.grader_id, {})
        if isinstance(entry, (int, float)):
            return float(entry) > 0.0
        return float(entry.get("weight", 0.0)) > 0.0

    def get_weight(self, config: dict[str, Any]) -> float:
        """Return this grader's weight for the current scenario (0.0–1.0)."""
        graders_cfg = config.get("graders", {})
        entry = graders_cfg.get(self.grader_id, {})
        if isinstance(entry, (int, float)):
            return float(entry)
        return float(entry.get("weight", 0.0))

    # ---------------------------------------------------------------------------
    # Shared delta-scoring helpers (used by multiple graders)
    # ---------------------------------------------------------------------------

    @staticmethod
    def delta_score(
        baseline_value: float,
        current_value: float,
        lower_is_better: bool = True,
    ) -> tuple[float, bool]:
        """
        Compute a [0.0, 1.0] score from a baseline→current delta.

        The most common scoring pattern: how much of the "problem" has been
        eliminated from the baseline level?

        Parameters
        ----------
        baseline_value  : The metric value at reset() (the "problem size").
        current_value   : The metric value at this step.
        lower_is_better : If True (default), reduction is improvement.
                          If False, increase is improvement.

        Returns
        -------
        (score, is_regression)
          score        : Proportional improvement in [0.0, 1.0].
                         = 1.0 when current reaches 0 (lower_is_better=True)
                         = 0.0 when current == baseline or regresses
          is_regression: True if current is worse than baseline.

        Examples
        --------
        baseline=20 violations, current=15 → score=0.25, not regression
        baseline=20 violations, current=0  → score=1.0,  not regression
        baseline=20 violations, current=25 → score=0.0,  is_regression=True
        baseline=0  violations, current=0  → score=1.0  (already at gold)
        """
        if lower_is_better:
            if baseline_value <= 0:
                # Already at gold standard at baseline
                return (1.0, False)
            improvement = baseline_value - current_value
            is_regression = current_value > baseline_value
            score = _clamp(improvement / baseline_value)
        else:
            # Higher is better (e.g. coverage %)
            if baseline_value >= 1.0:
                return (1.0, False)
            improvement = current_value - baseline_value
            headroom = 1.0 - baseline_value
            is_regression = current_value < baseline_value
            score = _clamp(improvement / max(headroom, 1e-9))

        return (score, is_regression)

    @staticmethod
    def progress_toward_target(
        baseline_value: float,
        current_value: float,
        target_value: float,
        lower_is_better: bool = True,
    ) -> tuple[float, bool]:
        """
        Score progress toward a specific target (not necessarily 0).

        Used when the scenario defines a threshold (e.g. target_coverage=0.80)
        rather than a universal gold standard.

        Returns
        -------
        (score, is_regression)
          score = 1.0 when current_value reaches target_value.
          score = 0.0 when current_value == baseline_value.

        Examples
        --------
        baseline=0.40 coverage, target=0.80, current=0.60 → score=0.50
        baseline=20 violations, target=5, current=12     → score=0.53
        """
        if lower_is_better:
            gap_total = baseline_value - target_value
            gap_remaining = current_value - target_value
            if gap_total <= 0:
                return (1.0, False)
            is_regression = current_value > baseline_value
            score = _clamp(1.0 - gap_remaining / gap_total)
        else:
            gap_total = target_value - baseline_value
            gap_remaining = target_value - current_value
            if gap_total <= 0:
                return (1.0, False)
            is_regression = current_value < baseline_value
            score = _clamp(1.0 - gap_remaining / gap_total)

        return (score, is_regression)

    @staticmethod
    def checklist_score(
        checks: list[tuple[str, bool, float]],
    ) -> tuple[float, list[str]]:
        """
        Score a weighted boolean checklist.

        Used by StructureGrader and ProductionGrader where the "metric"
        is a set of pass/fail invariants, each with an optional weight.

        Parameters
        ----------
        checks : list of (label, passed: bool, weight: float)
                 All weights are normalized internally.

        Returns
        -------
        (score, failed_labels)
          score         : Weighted fraction of passed checks, in [0.0, 1.0].
          failed_labels : List of labels for checks that did not pass.

        Examples
        --------
        [("no_cycles", True, 1.0), ("required_file", False, 2.0)]
          → score = 1.0/(1.0+2.0) = 0.333, failed=["required_file"]
        """
        if not checks:
            return (1.0, [])

        total_weight = sum(w for _, _, w in checks)
        if total_weight <= 0:
            return (1.0, [])

        passed_weight = sum(w for _, ok, w in checks if ok)
        failed = [label for label, ok, _ in checks if not ok]
        score = _clamp(passed_weight / total_weight)
        return (score, failed)

    @staticmethod
    def ratio_score(numerator: float, denominator: float) -> float:
        """
        Simple ratio score: numerator / denominator, clamped to [0.0, 1.0].
        Returns 1.0 if denominator is 0 (vacuously true — nothing to measure).
        """
        if denominator <= 0:
            return 1.0
        return _clamp(numerator / denominator)

    @staticmethod
    def build_feedback(
        grader_name: str,
        baseline: dict[str, Any],
        current: dict[str, Any],
        score: float,
        key_metric: str,
        unit: str = "",
        extra: Optional[str] = None,
    ) -> str:
        """
        Build a standard one-line feedback string.

        Format:
          "[GraderName] key_metric: {baseline_val} → {current_val}{unit}
           ({score:.0%} improved). {extra}"

        Examples
        --------
        "[Lint] violations: 17 → 4 (76% improved). Top remaining: F401 ×2"
        "[Coverage] line_coverage: 52% → 71% (40% improved)."
        """
        b_val = baseline.get(key_metric, "?")
        c_val = current.get(key_metric, "?")

        # Format numeric values cleanly
        def _fmt(v: Any) -> str:
            if isinstance(v, float):
                if unit == "%":
                    return f"{v:.0%}"
                return f"{v:.2f}"
            return str(v)

        base_str = _fmt(b_val)
        curr_str = _fmt(c_val)
        pct = f"{score:.0%}"
        msg = f"[{grader_name}] {key_metric}: {base_str}{unit} → {curr_str}{unit} ({pct} improved)."
        if extra:
            msg += f" {extra}"
        return msg


# ---------------------------------------------------------------------------
# Module-level helpers (not part of the grader interface)
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float to [lo, hi]. Handles NaN by returning lo."""
    if math.isnan(value) or math.isinf(value):
        return lo
    return max(lo, min(hi, value))


def empty_grade(
    grader_name: str,
    reason: str = "no metrics available",
) -> GradeResult:
    """
    Return a neutral GradeResult when metrics cannot be computed.

    Used when a tool is unavailable, a file doesn't exist yet, or
    a scenario has weight=0 for this grader.
    """
    return GradeResult(
        score=0.0,
        gold_distance=1.0,
        raw_baseline={},
        raw_current={},
        delta={},
        feedback=f"[{grader_name}] {reason}",
        solved=False,
        is_regression=False,
    )


def already_gold(
    grader_name: str,
    metrics: dict[str, Any],
) -> GradeResult:
    """
    Return a perfect GradeResult when baseline is already at gold standard.

    Used when the baseline repo has zero violations / perfect coverage / etc.
    before the agent does anything — score stays 1.0 throughout the episode.
    """
    return GradeResult(
        score=1.0,
        gold_distance=0.0,
        raw_baseline=metrics,
        raw_current=metrics,
        delta={k: 0 for k in metrics},
        feedback=f"[{grader_name}] Already at gold standard.",
        solved=True,
        is_regression=False,
    )
