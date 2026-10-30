import numpy as np

import rollout_stats as rs


def test_type_one_error_controlled_under_continuous_peeking():
    rng = np.random.default_rng(0)
    rejections = 0
    reps, steps = 400, 300
    for _ in range(reps):
        seq = rs.SequentialComparison(alpha=0.05)
        a = rng.random(steps) < 0.5
        b = rng.random(steps) < 0.5
        for i in range(steps):
            seq.update(a[i], b[i])
            if seq.decision:
                rejections += 1
                break
    assert rejections / reps <= 0.06


def test_naive_peeking_is_what_we_fix():
    """The classical test, checked after every 5 pairs, rejects far too often."""
    from scipy import stats

    rng = np.random.default_rng(1)
    reps, rej = 300, 0
    for _ in range(reps):
        a = rng.random(200) < 0.5
        b = rng.random(200) < 0.5
        for n in range(10, 201, 5):
            ka, kb = a[:n].sum(), b[:n].sum()
            if stats.fisher_exact([[ka, n - ka], [kb, n - kb]])[1] < 0.05:
                rej += 1
                break
    assert rej / reps > 0.08


def test_detects_large_difference_quickly():
    out = rs.simulate_sequential(0.9, 0.4, max_pairs=150, sims=300)
    assert out["correct_decision_rate"] > 0.95 and out["wrong_decision_rate"] < 0.01
    assert out["median_pairs_to_decision"] < 40


def test_one_sided_and_p_value():
    seq = rs.SequentialComparison(alternative="greater")
    seq.extend([(1, 0)] * 30)
    assert seq.decision == "A better" and seq.p_value < 0.05
    seq2 = rs.SequentialComparison(alternative="greater")
    seq2.extend([(0, 1)] * 30)
    assert seq2.decision is None and seq2.p_value == 1.0
    assert "pairs" in seq.summary()


def test_confidence_sequence_covers_uniformly():
    rng = np.random.default_rng(2)
    p, misses, reps = 0.3, 0, 200
    for _ in range(reps):
        x = rng.random(200) < p
        lo, hi = rs.confidence_sequence(x, alpha=0.05, grid=1000)
        misses += np.any((lo > p) | (hi < p))
    assert misses / reps <= 0.07
    lo, hi = rs.confidence_sequence(rng.random(400) < p, grid=1000)
    assert np.all(np.diff(hi - lo) <= 1e-12) and hi[-1] - lo[-1] < 0.2
