"""rollout-stats: honest statistics for robot policy evaluations with few rollouts.

    import rollout_stats as rs

    rs.rate_ci(17, 25)                       # Wilson interval
    print(rs.compare(17, 25, 11, 25, "ours", "baseline"))
    rs.n_per_arm(0.8, 0.6)                   # rollouts per policy for 80 % power
    seq = rs.SequentialComparison()          # peek after every rollout, stay valid
"""

from .compare import Comparison, cmh_test, compare, compare_paired, holm, prob_better
from .intervals import Rate, diff_ci, paired_diff_ci, rate_ci
from .planning import expected_halfwidth, mde, n_for_halfwidth, n_pairs, n_per_arm, power, power_paired
from .sequential import SequentialComparison, confidence_sequence, simulate_sequential
from .table import Results

__version__ = "0.1.0"

__all__ = [
    "Comparison", "Rate", "Results", "SequentialComparison", "cmh_test", "compare", "compare_paired",
    "confidence_sequence", "diff_ci", "expected_halfwidth", "holm", "mde", "n_for_halfwidth", "n_pairs", "n_per_arm",
    "paired_diff_ci", "power", "power_paired", "prob_better", "rate_ci", "simulate_sequential", "__version__",
]
