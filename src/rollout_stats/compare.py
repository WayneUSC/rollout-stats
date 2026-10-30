"""Is policy A better than policy B? Unpaired and paired comparisons, Bayesian summary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from .intervals import diff_ci, paired_diff_ci, rate_ci

ALTERNATIVES = ("two-sided", "greater", "less")


def _trapz(y, x):
    return np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)


def prob_better(k1: int, n1: int, k2: int, n2: int, prior: Tuple[float, float] = (1.0, 1.0)) -> float:
    """Posterior P(p1 > p2) under independent Beta priors (default uniform).

    Computed by numerical integration of f₁(x)·F₂(x) on a fine grid, so it is
    deterministic (no Monte Carlo noise).
    """
    a0, b0 = prior
    x = np.linspace(0, 1, 20001)
    f1 = stats.beta.pdf(x, k1 + a0, n1 - k1 + b0)
    f1 = np.where(np.isfinite(f1), f1, 0.0)
    F2 = stats.beta.cdf(x, k2 + a0, n2 - k2 + b0)
    total = _trapz(f1, x)
    return float(np.clip(_trapz(f1 * F2, x) / total, 0.0, 1.0))


@dataclass
class Comparison:
    name_a: str
    k_a: int
    n_a: int
    name_b: str
    k_b: int
    n_b: int
    test: str
    alternative: str
    p_value: float
    diff: float
    ci: Tuple[float, float]
    conf: float
    prob_a_better: float
    paired: bool = False
    discordant: Optional[Tuple[int, int]] = None
    notes: List[str] = field(default_factory=list)

    @property
    def significant(self) -> bool:
        return self.p_value < 1 - self.conf

    def verdict(self) -> str:
        alpha = 1 - self.conf
        lo, hi = self.ci
        if self.p_value < alpha:
            better = self.name_a if self.diff > 0 else self.name_b
            return f"{better} is better (p={_fmt_p(self.p_value)})"
        if hi - lo > 0.4:
            return (f"inconclusive — the difference could be anywhere from {lo:+.0%} to {hi:+.0%}; "
                    f"far more rollouts are needed")
        return f"no significant difference (p={_fmt_p(self.p_value)}); difference within [{lo:+.0%}, {hi:+.0%}]"

    def to_dict(self) -> Dict:
        return {
            "a": {"name": self.name_a, "successes": self.k_a, "trials": self.n_a},
            "b": {"name": self.name_b, "successes": self.k_b, "trials": self.n_b},
            "test": self.test, "alternative": self.alternative, "p_value": self.p_value,
            "difference": self.diff, "ci": list(self.ci), "confidence": self.conf,
            "prob_a_better": self.prob_a_better, "paired": self.paired,
            "discordant": list(self.discordant) if self.discordant else None,
            "verdict": self.verdict(), "notes": self.notes,
        }

    def __str__(self) -> str:
        la, ha = rate_ci(self.k_a, self.n_a, self.conf)
        lb, hb = rate_ci(self.k_b, self.n_b, self.conf)
        lines = [
            f"{self.name_a}: {self.k_a}/{self.n_a} = {self.k_a / self.n_a:.1%}  [{la:.1%}, {ha:.1%}]",
            f"{self.name_b}: {self.k_b}/{self.n_b} = {self.k_b / self.n_b:.1%}  [{lb:.1%}, {hb:.1%}]",
            f"difference {self.diff:+.1%}  {self.conf:.0%} CI [{self.ci[0]:+.1%}, {self.ci[1]:+.1%}]",
            f"{self.test} ({self.alternative}): p = {_fmt_p(self.p_value)}"
            + (f"   discordant pairs {self.discordant[0]} vs {self.discordant[1]}" if self.discordant else ""),
            f"P({self.name_a} better | data, uniform prior) = {self.prob_a_better:.3f}",
            f"→ {self.verdict()}",
        ]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def _fmt_p(p: float) -> str:
    return f"{p:.3g}" if p >= 1e-4 else "<1e-4"


def compare(
    k_a: int, n_a: int, k_b: int, n_b: int,
    name_a: str = "A", name_b: str = "B",
    test: str = "boschloo", alternative: str = "two-sided", conf: float = 0.95,
) -> Comparison:
    """Compare two independent success rates.

    ``test``: ``boschloo`` (default; exact unconditional, uniformly more powerful than
    Fisher), ``fisher`` (exact conditional, conservative) or ``barnard``.
    ``alternative``: ``two-sided``, ``greater`` (A better) or ``less``.
    """
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}")
    table = np.array([[k_a, n_a - k_a], [k_b, n_b - k_b]])
    if test == "fisher":
        p = stats.fisher_exact(table, alternative=alternative)[1]
    elif test == "boschloo":
        p = stats.boschloo_exact(table, alternative=alternative).pvalue
    elif test == "barnard":
        p = stats.barnard_exact(table, alternative=alternative).pvalue
    else:
        raise ValueError("test must be boschloo, fisher or barnard")
    if not np.isfinite(p):  # degenerate tables (e.g. 0/n vs 0/n)
        p = 1.0
    comp = Comparison(name_a, k_a, n_a, name_b, k_b, n_b, test, alternative, float(min(1.0, p)),
                      k_a / n_a - k_b / n_b, diff_ci(k_a, n_a, k_b, n_b, conf), conf,
                      prob_better(k_a, n_a, k_b, n_b))
    if min(n_a, n_b) < 10:
        comp.notes.append("fewer than 10 rollouts per policy: only very large differences can be detected")
    return comp


def compare_paired(
    outcomes_a: Sequence, outcomes_b: Sequence,
    name_a: str = "A", name_b: str = "B", alternative: str = "two-sided", conf: float = 0.95,
) -> Comparison:
    """Compare two policies evaluated on the *same* initial conditions (pair i = same seed,
    object pose, scene). Exact McNemar test on discordant pairs; Newcombe paired CI.

    Pairing removes episode difficulty from the comparison and usually needs far fewer
    rollouts than an unpaired design.
    """
    a = np.asarray(outcomes_a, dtype=bool)
    b = np.asarray(outcomes_b, dtype=bool)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired outcomes must be two 1-D sequences of equal length")
    both = int(np.sum(a & b))
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    neither = int(np.sum(~a & ~b))
    n = len(a)
    disc = only_a + only_b
    if disc == 0:
        p = 1.0
    else:
        alt = {"two-sided": "two-sided", "greater": "greater", "less": "less"}[alternative]
        p = stats.binomtest(only_a, disc, 0.5, alternative=alt).pvalue
    comp = Comparison(name_a, int(a.sum()), n, name_b, int(b.sum()), n, "exact McNemar", alternative, float(p),
                      (only_a - only_b) / n, paired_diff_ci(both, only_a, only_b, neither, conf), conf,
                      prob_better(only_a, disc, only_b, disc) if disc else 0.5,
                      paired=True, discordant=(only_a, only_b))
    comp.notes.append("P(better) here is the posterior that A wins more discordant pairs than B")
    return comp


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm–Bonferroni adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def cmh_test(tables: Sequence[np.ndarray], correction: bool = True) -> Tuple[float, float, float]:
    """Cochran–Mantel–Haenszel test across strata (tasks) of 2×2 tables
    ``[[k_a, n_a − k_a], [k_b, n_b − k_b]]``.

    Returns (statistic, p-value, Mantel–Haenszel common odds ratio). Use it to pool a
    comparison over several tasks without pretending the tasks are one population.
    """
    num = var = 0.0
    or_num = or_den = 0.0
    for t in tables:
        t = np.asarray(t, dtype=float)
        n = t.sum()
        if n <= 1:
            continue
        r1, r2 = t[0].sum(), t[1].sum()
        c1, c2 = t[:, 0].sum(), t[:, 1].sum()
        num += t[0, 0] - r1 * c1 / n
        var += r1 * r2 * c1 * c2 / (n ** 2 * (n - 1))
        or_num += t[0, 0] * t[1, 1] / n
        or_den += t[0, 1] * t[1, 0] / n
    if var == 0:
        return 0.0, 1.0, float("nan")
    stat = (max(0.0, abs(num) - 0.5) if correction else abs(num)) ** 2 / var
    p = float(stats.chi2.sf(stat, 1))
    mh_or = or_num / or_den if or_den > 0 else float("inf")
    return float(stat), p, float(mh_or)
