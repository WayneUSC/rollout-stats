import numpy as np
import pytest
from scipy import stats

import rollout_stats as rs
from rollout_stats.planning import _fisher_reject


def exact_coverage(n, p, method, conf=0.95):
    k = np.arange(n + 1)
    pmf = stats.binom.pmf(k, n, p)
    cover = [lo <= p <= hi for lo, hi in (rs.rate_ci(int(j), n, conf, method) for j in k)]
    return float(np.sum(pmf * np.array(cover)))


def test_wilson_known_value():
    lo, hi = rs.rate_ci(17, 25)
    assert lo == pytest.approx(0.4841, abs=1e-4) and hi == pytest.approx(0.8279, abs=1e-4)
    assert rs.rate_ci(0, 20)[0] == 0.0 and rs.rate_ci(20, 20)[1] == 1.0
    assert rs.rate_ci(0, 20)[1] > 0.1  # zero successes is not zero uncertainty


@pytest.mark.parametrize("n", [10, 25, 50])
def test_clopper_pearson_is_exact_and_wilson_is_close(n):
    ps = np.linspace(0.02, 0.98, 49)
    cp = [exact_coverage(n, p, "clopper-pearson") for p in ps]
    wi = [exact_coverage(n, p, "wilson") for p in ps]
    assert min(cp) >= 0.95 - 1e-9
    assert 0.935 <= np.mean(wi) <= 0.965


def test_newcombe_unpaired_reference_example():
    # Newcombe (1998), Statistics in Medicine 17:873–890, example 56/70 vs 48/80, method 10
    lo, hi = rs.diff_ci(56, 70, 48, 80)
    assert lo == pytest.approx(0.0524, abs=5e-4) and hi == pytest.approx(0.3339, abs=5e-4)


def test_paired_ci_properties():
    lo, hi = rs.paired_diff_ci(20, 8, 2, 10)
    d = (8 - 2) / 40
    assert lo < d < hi and -1 <= lo and hi <= 1
    lo0, hi0 = rs.paired_diff_ci(10, 0, 0, 10)
    assert lo0 < 0 < hi0


def test_paired_ci_coverage_by_simulation():
    rng = np.random.default_rng(0)
    n, hits, reps = 30, 0, 600
    p11, p10, p01 = 0.4, 0.25, 0.1  # true diff = 0.15
    for _ in range(reps):
        cells = rng.multinomial(n, [p11, p10, p01, 1 - p11 - p10 - p01])
        lo, hi = rs.paired_diff_ci(*cells)
        hits += lo <= 0.15 <= hi
    assert hits / reps >= 0.92


def test_fisher_reject_matches_scipy():
    for n_a, n_b in [(6, 6), (8, 5)]:
        for alt in ("two-sided", "greater", "less"):
            rej = _fisher_reject(n_a, n_b, 0.05, alt)
            for ka in range(n_a + 1):
                for kb in range(n_b + 1):
                    p = stats.fisher_exact([[ka, n_a - ka], [kb, n_b - kb]], alternative=alt)[1]
                    assert rej[ka, kb] == (p <= 0.05 + 1e-12), (n_a, n_b, alt, ka, kb, p)


def test_power_and_planning_are_consistent():
    n = rs.n_per_arm(0.8, 0.6)
    assert rs.power(0.8, 0.6, n) >= 0.8 > rs.power(0.8, 0.6, n - 1)
    assert 80 <= n <= 100
    assert rs.power(0.5, 0.5, 30) <= 0.05  # exact test is conservative under the null
    assert rs.n_for_halfwidth(0.7, 0.1) == pytest.approx(77, abs=3)
    d = rs.mde(0.5, 25)
    assert rs.power(0.5 + d, 0.5, 25) >= 0.8


def test_paired_power_beats_unpaired_when_outcomes_correlate():
    # both policies fail the same hard episodes: only 25 % of pairs are discordant
    unpaired = rs.power(0.7, 0.5, 40)
    paired = rs.power_paired(0.225, 0.025, 40)  # p_a = 0.7, p_b = 0.5 with strong correlation
    assert paired > unpaired + 0.2


def test_compare_and_prob_better():
    c = rs.compare(17, 25, 11, 25, test="fisher")
    assert c.p_value == pytest.approx(stats.fisher_exact([[17, 8], [11, 14]])[1])
    assert rs.prob_better(10, 20, 10, 20) == pytest.approx(0.5, abs=1e-6)
    rng = np.random.default_rng(0)
    mc = np.mean(rng.beta(18, 9, 200000) > rng.beta(12, 15, 200000))
    assert rs.prob_better(17, 25, 11, 25) == pytest.approx(mc, abs=0.005)
    assert "inconclusive" in rs.compare(3, 5, 1, 5).verdict()
    assert rs.compare(0, 10, 0, 10).p_value == 1.0


def test_paired_mcnemar():
    a = [1] * 12 + [0] * 3 + [1] * 10 + [0] * 5
    b = [0] * 12 + [1] * 3 + [1] * 10 + [0] * 5
    c = rs.compare_paired(a, b)
    assert c.discordant == (12, 3)
    assert c.p_value == pytest.approx(stats.binomtest(12, 15, 0.5).pvalue)


def test_cmh_single_stratum_matches_pearson():
    t = np.array([[17, 8], [11, 14]])
    stat, p, _ = rs.cmh_test([t], correction=False)
    chi2 = stats.chi2_contingency(t, correction=False)[0]
    assert stat == pytest.approx(chi2 * (t.sum() - 1) / t.sum())


def test_holm():
    assert np.allclose(rs.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
