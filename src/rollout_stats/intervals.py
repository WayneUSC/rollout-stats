"""Confidence intervals for a success rate and for differences between success rates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import stats

METHODS = ("wilson", "clopper-pearson", "jeffreys", "agresti-coull")


def _check(k: int, n: int) -> None:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError(f"successes must be between 0 and n (got {k} of {n})")


def rate_ci(k: int, n: int, conf: float = 0.95, method: str = "wilson") -> Tuple[float, float]:
    """Confidence interval for a success rate k/n.

    ``wilson`` (default) has close-to-nominal coverage even for 10–30 rollouts and never
    leaves [0, 1]. ``clopper-pearson`` is exact and conservative. ``jeffreys`` is the
    equal-tailed Beta(k+½, n−k+½) interval. The Wald interval (p ± 1.96·SE) is
    deliberately not offered: at 0/20 or 20/20 it has zero width.
    """
    _check(k, n)
    alpha = 1 - conf
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    if method == "wilson":
        denom = 1 + z ** 2 / n
        center = (p + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
        lo, hi = center - half, center + half
        # exact boundary values at k = 0 or n
        lo = 0.0 if k == 0 else lo
        hi = 1.0 if k == n else hi
        return float(max(0.0, lo)), float(min(1.0, hi))
    if method == "clopper-pearson":
        lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
        hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
        return float(lo), float(hi)
    if method == "jeffreys":
        lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k + 0.5, n - k + 0.5)
        hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5)
        return float(lo), float(hi)
    if method == "agresti-coull":
        nt = n + z ** 2
        pt = (k + z ** 2 / 2) / nt
        half = z * np.sqrt(pt * (1 - pt) / nt)
        return float(max(0.0, pt - half)), float(min(1.0, pt + half))
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def diff_ci(k1: int, n1: int, k2: int, n2: int, conf: float = 0.95) -> Tuple[float, float]:
    """Newcombe (1998, method 10) hybrid-score interval for p1 − p2, independent samples.

    Built from the two Wilson intervals; good coverage at small n, unlike the Wald
    difference interval.
    """
    _check(k1, n1)
    _check(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = rate_ci(k1, n1, conf)
    l2, u2 = rate_ci(k2, n2, conf)
    d = p1 - p2
    lo = d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return float(max(-1.0, lo)), float(min(1.0, hi))


def paired_diff_ci(a: int, b: int, c: int, d: int, conf: float = 0.95) -> Tuple[float, float]:
    """Newcombe (1998, method 10) interval for p_A − p_B from *paired* rollouts.

    a = both succeed, b = only A succeeds, c = only B succeeds, d = both fail — e.g. both
    policies evaluated from the same initial conditions / seeds.
    """
    n = a + b + c + d
    if n <= 0:
        raise ValueError("no pairs")
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = rate_ci(a + b, n, conf)
    l2, u2 = rate_ci(a + c, n, conf)
    denom = np.sqrt(float((a + b) * (c + d) * (a + c) * (b + d)))
    if denom == 0:
        phi = 0.0
    else:
        num = a * d - b * c
        phi = (max(num - n / 2, 0) if num > 0 else num) / denom
    dl = np.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = np.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    diff = p1 - p2
    return float(max(-1.0, diff - dl)), float(min(1.0, diff + du))


@dataclass
class Rate:
    successes: int
    trials: int
    conf: float = 0.95
    method: str = "wilson"

    @property
    def p(self) -> float:
        return self.successes / self.trials

    @property
    def ci(self) -> Tuple[float, float]:
        return rate_ci(self.successes, self.trials, self.conf, self.method)

    def __str__(self) -> str:
        lo, hi = self.ci
        return f"{self.successes}/{self.trials} = {self.p:.1%} [{lo:.1%}, {hi:.1%}]"

    def latex(self, digits: int = 0) -> str:
        lo, hi = self.ci
        f = f"{{:.{digits}f}}"
        return f"{f.format(100 * self.p)}\\% {{\\scriptsize [{f.format(100 * lo)}, {f.format(100 * hi)}]}}"
