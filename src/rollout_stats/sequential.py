"""Anytime-valid inference: look after every rollout, stop whenever you like.

Classical p-values assume the number of rollouts was fixed in advance. Checking after
every batch and stopping "once it is significant" inflates the false-positive rate far
above 5 %. The tests here are built on nonnegative supermartingales ("testing by
betting", Shafer 2021; Waudby-Smith & Ramdas 2023): by Ville's inequality the chance
that the wealth *ever* reaches 1/α under the null is at most α, no matter when you
look or stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


class _Bettor:
    """Wealth process for H0: E[D] ≤ 0 with D ∈ [−1, 1] (aGRAPA-style plug-in bets)."""

    def __init__(self, max_bet: float = 0.5, prior_n: float = 1.0):
        self.max_bet = max_bet
        self.prior_n = prior_n
        self.sum = 0.0
        self.sumsq = 0.0
        self.n = 0
        self.log_wealth = 0.0
        self.max_log_wealth = 0.0

    def bet(self) -> float:
        n_eff = self.prior_n + self.n
        mu = self.sum / n_eff
        var = (0.25 * self.prior_n + self.sumsq - n_eff * mu ** 2) / n_eff
        var = max(var, 1e-3)
        lam = mu / (var + mu ** 2) if mu > 0 else 0.0
        return float(np.clip(lam, 0.0, self.max_bet))

    def update(self, d: float) -> None:
        lam = self.bet()
        self.log_wealth += np.log1p(lam * d)
        self.max_log_wealth = max(self.max_log_wealth, self.log_wealth)
        self.sum += d
        self.sumsq += d * d
        self.n += 1


@dataclass
class SequentialComparison:
    """Anytime-valid comparison of two policies run in pairs (A, B, A, B, ...).

    Each update takes one rollout of A and one of B (ideally from the same initial
    condition, interleaved in time so drifts in the setup hit both equally). The test
    stops with a decision as soon as the evidence is sufficient — often far earlier than
    a fixed-n design when the difference is large — and never needs a correction for
    peeking.

    >>> seq = SequentialComparison(alpha=0.05)
    >>> for a, b in rollouts:
    ...     seq.update(a, b)
    ...     if seq.decision: break
    """

    alpha: float = 0.05
    alternative: str = "two-sided"  # or "greater" (A better) / "less" (B better)
    max_bet: float = 0.5
    history: List[Tuple[int, int]] = field(default_factory=list)
    decision: Optional[str] = None
    decided_at: Optional[int] = None

    def __post_init__(self) -> None:
        if self.alternative not in ("two-sided", "greater", "less"):
            raise ValueError("alternative must be two-sided, greater or less")
        self._up = _Bettor(self.max_bet)    # evidence that A > B
        self._down = _Bettor(self.max_bet)  # evidence that B > A
        self.wealth_trace: List[Tuple[float, float]] = []

    @property
    def _level(self) -> float:
        return self.alpha / 2 if self.alternative == "two-sided" else self.alpha

    @property
    def n(self) -> int:
        return len(self.history)

    def update(self, a, b) -> "SequentialComparison":
        a, b = int(bool(a)), int(bool(b))
        d = float(a - b)
        self.history.append((a, b))
        self._up.update(d)
        self._down.update(-d)
        self.wealth_trace.append((float(np.exp(self._up.log_wealth)), float(np.exp(self._down.log_wealth))))
        if self.decision is None:
            thr = np.log(1.0 / self._level)
            if self.alternative in ("two-sided", "greater") and self._up.log_wealth >= thr:
                self.decision, self.decided_at = "A better", self.n
            elif self.alternative in ("two-sided", "less") and self._down.log_wealth >= thr:
                self.decision, self.decided_at = "B better", self.n
        return self

    def extend(self, pairs: Sequence[Tuple[int, int]]) -> "SequentialComparison":
        for a, b in pairs:
            self.update(a, b)
        return self

    @property
    def p_value(self) -> float:
        """Anytime-valid p-value: valid at whatever moment you read it."""
        p_up = float(min(1.0, np.exp(-self._up.max_log_wealth)))
        p_down = float(min(1.0, np.exp(-self._down.max_log_wealth)))
        if self.alternative == "greater":
            return p_up
        if self.alternative == "less":
            return p_down
        return float(min(1.0, 2 * min(p_up, p_down)))

    def summary(self) -> str:
        ka = sum(a for a, _ in self.history)
        kb = sum(b for _, b in self.history)
        n = self.n
        wa, wb = self.wealth_trace[-1] if self.wealth_trace else (1.0, 1.0)
        state = (f"DECISION after {self.decided_at} pairs: {self.decision}" if self.decision
                 else "continue — not enough evidence yet")
        return (f"{n} pairs: A {ka}/{n}, B {kb}/{n}; wealth A>B {wa:.2f}, B>A {wb:.2f} "
                f"(threshold {1 / self._level:.0f}); anytime-valid p = {self.p_value:.3g}\n{state}")


def confidence_sequence(outcomes: Sequence, alpha: float = 0.05, grid: int = 2000,
                        max_bet: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Anytime-valid confidence sequence for a success rate (hedged capital, Waudby-Smith
    & Ramdas 2023). Returns arrays ``lo[t], hi[t]`` after each rollout t; the true rate is
    inside *all* of them simultaneously with probability ≥ 1 − α.
    """
    x = np.asarray(outcomes, dtype=float)
    m = (np.arange(grid) + 0.5) / grid
    log_kp = np.zeros(grid)
    log_km = np.zeros(grid)
    thr = np.log(1.0 / alpha)
    lo = np.empty(len(x))
    hi = np.empty(len(x))
    run_lo, run_hi = 0.0, 1.0
    s = 0.0
    ss = 0.25
    for t, xt in enumerate(x, start=1):
        var_prev = ss / t
        lam = np.sqrt(2 * np.log(2 / alpha) / (var_prev * t * np.log(t + 1)))
        lp = np.minimum(lam, max_bet / m)
        lm = np.minimum(lam, max_bet / (1 - m))
        log_kp += np.log1p(lp * (xt - m))
        log_km += np.log1p(-lm * (xt - m))
        hedged = np.maximum(log_kp + np.log(0.5), log_km + np.log(0.5))
        keep = hedged < thr
        if keep.any():
            run_lo = max(run_lo, float(m[keep].min() - 0.5 / grid))
            run_hi = min(run_hi, float(m[keep].max() + 0.5 / grid))
        lo[t - 1], hi[t - 1] = max(0.0, run_lo), min(1.0, run_hi)
        s += xt
        mu = (0.5 + s) / (t + 1)
        ss += (xt - mu) ** 2
    return lo, hi


def simulate_sequential(p_a: float, p_b: float, alpha: float = 0.05, max_pairs: int = 200, sims: int = 2000,
                        seed: int = 0, alternative: str = "two-sided") -> dict:
    """Monte Carlo: how often and how quickly the sequential test decides."""
    rng = np.random.default_rng(seed)
    stops, correct, wrong = [], 0, 0
    for _ in range(sims):
        seq = SequentialComparison(alpha=alpha, alternative=alternative)
        a = rng.random(max_pairs) < p_a
        b = rng.random(max_pairs) < p_b
        for i in range(max_pairs):
            seq.update(a[i], b[i])
            if seq.decision:
                break
        if seq.decision:
            stops.append(seq.decided_at)
            truth = "A better" if p_a > p_b else ("B better" if p_b > p_a else None)
            if seq.decision == truth:
                correct += 1
            else:
                wrong += 1
    return {
        "p_a": p_a, "p_b": p_b, "alpha": alpha, "max_pairs": max_pairs, "sims": sims,
        "decided": len(stops) / sims,
        "correct_decision_rate": correct / sims,
        "wrong_decision_rate": wrong / sims,
        "median_pairs_to_decision": float(np.median(stops)) if stops else None,
        "p90_pairs_to_decision": float(np.quantile(stops, 0.9)) if stops else None,
    }
