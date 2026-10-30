"""How many rollouts do I need? Exact power and precision calculations."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats


def _two_sided_pvalues(pmf: np.ndarray) -> np.ndarray:
    """Exact two-sided p-value for every outcome: total probability of outcomes no more
    likely than it (the 'minimum likelihood' rule used by Fisher's test)."""
    order = np.sort(pmf)
    csum = np.cumsum(order)
    idx = np.searchsorted(order, pmf * (1 + 1e-7), side="right")
    return np.minimum(1.0, csum[idx - 1])


def _wilson_halfwidth(k: np.ndarray, n: int, conf: float) -> np.ndarray:
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z ** 2 / n
    return z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom


def expected_halfwidth(p: float, n: int, conf: float = 0.95) -> float:
    """Expected half-width of the Wilson interval when the true rate is ``p``."""
    k = np.arange(n + 1)
    return float(np.sum(stats.binom.pmf(k, n, p) * _wilson_halfwidth(k, n, conf)))


def n_for_halfwidth(p: float, halfwidth: float, conf: float = 0.95, n_max: int = 100000) -> int:
    """Smallest n whose *expected* Wilson half-width is ≤ ``halfwidth``.

    Example: to report a ~70 % success rate to within ±10 points you need about 77
    rollouts, not 20.
    """
    if not 0 < halfwidth < 0.5:
        raise ValueError("halfwidth must be in (0, 0.5)")
    lo, hi = 1, 2
    while expected_halfwidth(p, hi, conf) > halfwidth:
        lo, hi = hi, hi * 2
        if hi > n_max:
            raise ValueError("n exceeds n_max")
    while lo < hi:
        mid = (lo + hi) // 2
        if expected_halfwidth(p, mid, conf) > halfwidth:
            lo = mid + 1
        else:
            hi = mid
    return int(lo)


def _fisher_reject(n_a: int, n_b: int, alpha: float, alternative: str) -> np.ndarray:
    """Boolean (n_a+1, n_b+1) matrix: does Fisher's exact test reject at (k_a, k_b)?

    The conditional (hypergeometric) probabilities of every table are computed at once
    from log-binomial coefficients; each anti-diagonal k_a + k_b = s is one conditional
    distribution.
    """
    from scipy.special import gammaln

    def lchoose(n, k):
        return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

    N = n_a + n_b
    ka = np.arange(n_a + 1)[:, None]
    kb = np.arange(n_b + 1)[None, :]
    logp = lchoose(n_a, ka) + lchoose(n_b, kb) - lchoose(N, ka + kb)
    pmf_all = np.exp(logp)
    reject = np.zeros((n_a + 1, n_b + 1), dtype=bool)
    for s in range(N + 1):
        a = np.arange(max(0, s - n_b), min(n_a, s) + 1)
        pmf = pmf_all[a, s - a]
        if alternative == "greater":
            p = np.cumsum(pmf[::-1])[::-1]
        elif alternative == "less":
            p = np.cumsum(pmf)
        else:
            p = _two_sided_pvalues(pmf)
        reject[a, s - a] = p <= alpha * (1 + 1e-9)
    return reject


def power(p_a: float, p_b: float, n_a: int, n_b: Optional[int] = None, alpha: float = 0.05,
          alternative: str = "two-sided") -> float:
    """Exact power of Fisher's exact test for n_a vs n_b independent rollouts.

    Computed by enumerating every possible outcome, not by a normal approximation
    (which is badly off at the sample sizes robotics papers use). Boschloo's test, the
    default in :func:`rollout_stats.compare`, is slightly more powerful, so this is a
    safe lower bound.
    """
    n_b = n_a if n_b is None else n_b
    rej = _fisher_reject(n_a, n_b, alpha, alternative)
    pa = stats.binom.pmf(np.arange(n_a + 1), n_a, p_a)
    pb = stats.binom.pmf(np.arange(n_b + 1), n_b, p_b)
    return float(pa @ rej.astype(float) @ pb)


def n_per_arm(p_a: float, p_b: float, target_power: float = 0.8, alpha: float = 0.05,
              alternative: str = "two-sided", n_max: int = 5000) -> int:
    """Smallest n per policy reaching ``target_power`` (exact Fisher power).

    Exact power is slightly saw-toothed in n; the search brackets the answer by doubling,
    bisects, then confirms that n − 1 falls short.
    """
    if p_a == p_b:
        raise ValueError("p_a and p_b are equal: there is no difference to detect")

    def ok(n: int) -> bool:
        return power(p_a, p_b, n, n, alpha, alternative) >= target_power

    lo, hi = 1, 2
    while not ok(hi):
        lo, hi = hi, hi * 2
        if hi > n_max:
            if ok(n_max):
                hi = n_max
                break
            raise ValueError(f"more than {n_max} rollouts per policy needed")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    while hi > 2 and ok(hi - 1):
        hi -= 1
    return int(hi)


def power_paired(p_only_a: float, p_only_b: float, n: int, alpha: float = 0.05,
                 alternative: str = "two-sided") -> float:
    """Exact power of the McNemar test for n paired rollouts.

    ``p_only_a`` / ``p_only_b``: probability that a pair is won by A only / B only.
    Pairs both succeed or both fail on carry no information — which is why pairing
    helps so much when most of the variation comes from episode difficulty.
    """
    p_disc = p_only_a + p_only_b
    if not 0 < p_disc <= 1:
        raise ValueError("p_only_a + p_only_b must be in (0, 1]")
    q = p_only_a / p_disc
    total = 0.0
    for d in range(1, n + 1):
        pd = stats.binom.pmf(d, n, p_disc)
        if pd < 1e-15:
            continue
        k = np.arange(d + 1)
        if alternative == "greater":
            pv = stats.binom.sf(k - 1, d, 0.5)
        elif alternative == "less":
            pv = stats.binom.cdf(k, d, 0.5)
        else:
            pv = _two_sided_pvalues(stats.binom.pmf(k, d, 0.5))
        total += pd * float(np.sum(stats.binom.pmf(k, d, q) * (pv <= alpha)))
    return total


def n_pairs(p_only_a: float, p_only_b: float, target_power: float = 0.8, alpha: float = 0.05,
            alternative: str = "two-sided", n_max: int = 5000) -> int:
    """Number of paired rollouts reaching ``target_power`` with the exact McNemar test."""
    if p_only_a == p_only_b:
        raise ValueError("no difference between discordant rates")

    def ok(n: int) -> bool:
        return power_paired(p_only_a, p_only_b, n, alpha, alternative) >= target_power

    lo, hi = 1, 2
    while not ok(hi):
        lo, hi = hi, hi * 2
        if hi > n_max:
            raise ValueError(f"more than {n_max} pairs needed")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return int(hi)


def mde(p_b: float, n: int, target_power: float = 0.8, alpha: float = 0.05, alternative: str = "two-sided") -> float:
    """Minimum detectable improvement over a baseline rate ``p_b`` with n rollouts each."""
    lo, hi = p_b, 1.0
    if power(1.0, p_b, n, n, alpha, alternative) < target_power:
        return float("nan")
    for _ in range(30):
        mid = (lo + hi) / 2
        if power(mid, p_b, n, n, alpha, alternative) >= target_power:
            hi = mid
        else:
            lo = mid
    return float(hi - p_b)
