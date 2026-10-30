"""Command-line interface for rollout-stats."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import pandas as pd

from . import __version__
from .compare import compare, compare_paired
from .intervals import METHODS, rate_ci
from .planning import expected_halfwidth, mde, n_for_halfwidth, n_per_arm, power
from .sequential import SequentialComparison, simulate_sequential
from .table import Results


def _frac(s: str):
    try:
        k, n = s.split("/")
        return int(k), int(n)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected SUCCESSES/TRIALS like 17/25, got {s!r}")


def cmd_ci(a) -> int:
    lo, hi = rate_ci(a.successes, a.trials, a.conf, a.method)
    p = a.successes / a.trials
    print(f"{a.successes}/{a.trials} = {p:.1%}   {a.conf:.0%} {a.method} CI [{lo:.1%}, {hi:.1%}]   (±{(hi - lo) / 2:.1%})")
    if a.trials < 50:
        need = n_for_halfwidth(min(max(p, 0.02), 0.98), 0.05, a.conf)
        print(f"for ±5 points at this rate you would need about {need} rollouts")
    return 0


def cmd_compare(a) -> int:
    names = (a.names.split(",") + ["A", "B"])[:2] if a.names else ["A", "B"]
    (ka, na), (kb, nb) = a.a, a.b
    c = compare(ka, na, kb, nb, names[0], names[1], test=a.test, alternative=a.alternative, conf=a.conf)
    print(c)
    if not c.significant and ka / na != kb / nb:
        try:
            n = n_per_arm(min(max(ka / na, .01), .99), min(max(kb / nb, .01), .99), 0.8, 1 - a.conf, a.alternative)
            print(f"\nIf these rates are the truth, ~{n} rollouts per policy give 80% power to detect the difference.")
        except ValueError:
            pass
    return 0


def cmd_paired(a) -> int:
    df = pd.read_csv(a.csv, sep=None, engine="python")
    for col in (a.a, a.b):
        if col not in df.columns:
            raise ValueError(f"column {col!r} not in {list(df.columns)}")
    print(compare_paired(df[a.a].astype(int), df[a.b].astype(int), a.a, a.b, a.alternative, a.conf))
    return 0


def cmd_plan(a) -> int:
    alpha = 1 - a.conf
    if a.halfwidth is not None:
        if a.p is None:
            raise ValueError("--halfwidth needs --p (expected success rate)")
        n = n_for_halfwidth(a.p, a.halfwidth, a.conf)
        print(f"To estimate a ~{a.p:.0%} success rate to ±{a.halfwidth:.0%} ({a.conf:.0%} Wilson): {n} rollouts.")
        for m in (10, 20, 50, 100):
            print(f"  with {m:>3} rollouts the interval is about ±{expected_halfwidth(a.p, m, a.conf):.0%}")
        return 0
    if a.p_a is not None and a.p_b is not None:
        if a.n:
            pw = power(a.p_a, a.p_b, a.n, a.n, alpha, a.alternative)
            print(f"Power to detect {a.p_a:.0%} vs {a.p_b:.0%} with {a.n} rollouts each: {pw:.0%} (exact, Fisher).")
        else:
            n = n_per_arm(a.p_a, a.p_b, a.power, alpha, a.alternative)
            print(f"To detect {a.p_a:.0%} vs {a.p_b:.0%} with {a.power:.0%} power (α={alpha:g}, {a.alternative}): "
                  f"{n} rollouts per policy ({2 * n} total).")
            seq = simulate_sequential(a.p_a, a.p_b, alpha, max_pairs=max(2 * n, 50), sims=a.sims,
                                      alternative=a.alternative)
            if seq["median_pairs_to_decision"]:
                print(f"Sequential alternative (anytime-valid, paired A/B rollouts, cap {seq['max_pairs']} pairs): "
                      f"decides in {seq['decided']:.0%} of runs, median {seq['median_pairs_to_decision']:.0f} pairs.")
        return 0
    if a.p_b is not None and a.n:
        d = mde(a.p_b, a.n, a.power, alpha, a.alternative)
        if d != d:
            print(f"With {a.n} rollouts each, not even a perfect policy is detectable over a {a.p_b:.0%} baseline.")
        else:
            print(f"With {a.n} rollouts each and a {a.p_b:.0%} baseline, the smallest improvement detectable with "
                  f"{a.power:.0%} power is +{d:.0%} (to {a.p_b + d:.0%}).")
        return 0
    raise ValueError("give --p-a/--p-b, --p-b with --n, or --p with --halfwidth")


def cmd_table(a) -> int:
    res = Results.from_csv(a.csv, policy=a.policy, task=a.task, pair_by=a.pair_by, conf=a.conf, method=a.method)
    kw = {"test": a.test}
    md = res.markdown(a.baseline, **kw) if a.baseline else res.markdown()
    print(md)
    if a.markdown:
        with open(a.markdown, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
    if a.latex:
        with open(a.latex, "w", encoding="utf-8") as fh:
            fh.write(res.latex() + "\n")
        print(f"\nLaTeX table: {a.latex}")
    if a.html:
        from .report import render_html

        with open(a.html, "w", encoding="utf-8") as fh:
            fh.write(render_html(res, a.baseline, title=a.title, **kw))
        print(f"HTML report: {a.html}")
    return 0


def cmd_sequential(a) -> int:
    df = pd.read_csv(a.csv, sep=None, engine="python")
    seq = SequentialComparison(alpha=1 - a.conf, alternative=a.alternative)
    for i, (x, y) in enumerate(zip(df[a.a].astype(int), df[a.b].astype(int)), start=1):
        seq.update(x, y)
        if a.verbose:
            wa, wb = seq.wealth_trace[-1]
            print(f"{i:4d}  A={x} B={y}  wealth A>B {wa:8.2f}  B>A {wb:8.2f}")
        if seq.decision and a.stop:
            break
    print(seq.summary())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rollout-stats",
                                description="Honest statistics for robot policy evaluations with few rollouts.")
    p.add_argument("--version", action="version", version=f"rollout-stats {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("ci", help="confidence interval for one success rate")
    c.add_argument("successes", type=int)
    c.add_argument("trials", type=int)
    c.add_argument("--method", default="wilson", choices=METHODS)
    c.add_argument("--conf", type=float, default=0.95)
    c.set_defaults(func=cmd_ci)

    m = sub.add_parser("compare", help="is A better than B? (independent rollouts)")
    m.add_argument("a", type=_frac, help="A as SUCCESSES/TRIALS, e.g. 17/25")
    m.add_argument("b", type=_frac, help="B as SUCCESSES/TRIALS")
    m.add_argument("--names", help="comma-separated names, e.g. ours,baseline")
    m.add_argument("--test", default="boschloo", choices=["boschloo", "fisher", "barnard"])
    m.add_argument("--alternative", default="two-sided", choices=["two-sided", "greater", "less"])
    m.add_argument("--conf", type=float, default=0.95)
    m.set_defaults(func=cmd_compare)

    pr = sub.add_parser("paired", help="compare two policies run on the same initial conditions (CSV with two 0/1 columns)")
    pr.add_argument("csv")
    pr.add_argument("--a", required=True, help="column with A's outcomes")
    pr.add_argument("--b", required=True, help="column with B's outcomes")
    pr.add_argument("--alternative", default="two-sided", choices=["two-sided", "greater", "less"])
    pr.add_argument("--conf", type=float, default=0.95)
    pr.set_defaults(func=cmd_paired)

    pl = sub.add_parser("plan", help="how many rollouts do I need?")
    pl.add_argument("--p-a", type=float, help="expected success rate of the new policy")
    pl.add_argument("--p-b", type=float, help="expected success rate of the baseline")
    pl.add_argument("--n", type=int, help="rollouts per policy you can afford")
    pl.add_argument("--p", type=float, help="expected success rate (for --halfwidth)")
    pl.add_argument("--halfwidth", type=float, help="target ± precision for one success rate, e.g. 0.1")
    pl.add_argument("--power", type=float, default=0.8)
    pl.add_argument("--alternative", default="two-sided", choices=["two-sided", "greater", "less"])
    pl.add_argument("--conf", type=float, default=0.95)
    pl.add_argument("--sims", type=int, default=1000, help="simulations for the sequential estimate")
    pl.set_defaults(func=cmd_plan)

    t = sub.add_parser("table", help="results table (Markdown/LaTeX/HTML) with comparisons against a baseline")
    t.add_argument("csv", help="one row per rollout (policy, task, success[, seed]) or per policy×task (successes, trials)")
    t.add_argument("--baseline")
    t.add_argument("--policy", default="policy")
    t.add_argument("--task", default="task")
    t.add_argument("--pair-by", help="column identifying shared initial conditions (seed, scene id) for paired tests")
    t.add_argument("--test", default="boschloo", choices=["boschloo", "fisher", "barnard"])
    t.add_argument("--method", default="wilson", choices=METHODS)
    t.add_argument("--conf", type=float, default=0.95)
    t.add_argument("--markdown")
    t.add_argument("--latex")
    t.add_argument("--html")
    t.add_argument("--title", default="Policy evaluation")
    t.set_defaults(func=cmd_table)

    s = sub.add_parser("sequential", help="anytime-valid A/B test over paired rollouts, in order")
    s.add_argument("csv")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--alternative", default="two-sided", choices=["two-sided", "greater", "less"])
    s.add_argument("--conf", type=float, default=0.95)
    s.add_argument("--stop", action="store_true", help="stop reading at the first decision")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_sequential)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"rollout-stats: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
