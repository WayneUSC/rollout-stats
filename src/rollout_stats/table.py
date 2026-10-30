"""Results tables for papers: success rates with intervals, comparisons against a baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .compare import cmh_test, compare, compare_paired, holm
from .intervals import rate_ci
from .planning import n_pairs, n_per_arm

ALL = "all tasks"


@dataclass
class Results:
    """Rollout outcomes for several policies (and optionally several tasks).

    Accepts either one row per rollout (``success`` column, 0/1 or bool, optionally a
    ``pair_by`` column such as seed / initial condition) or one row per policy×task
    (``successes`` and ``trials`` columns).
    """

    df: pd.DataFrame
    policy: str = "policy"
    task: Optional[str] = "task"
    pair_by: Optional[str] = None
    conf: float = 0.95
    method: str = "wilson"

    def __post_init__(self) -> None:
        df = self.df.copy()
        if self.policy not in df.columns:
            raise ValueError(f"no {self.policy!r} column (columns: {list(df.columns)})")
        if self.task is None or self.task not in df.columns:
            self.task = "task"
            df["task"] = ALL
        df[self.policy] = df[self.policy].astype(str)
        df[self.task] = df[self.task].astype(str)
        if "successes" in df.columns and "trials" in df.columns:
            self.long = None
            agg = df.groupby([self.policy, self.task], sort=False)[["successes", "trials"]].sum().reset_index()
            if self.pair_by:
                raise ValueError("pair_by needs one row per rollout (a 'success' column)")
        elif "success" in df.columns:
            s = df["success"]
            if s.dtype == object:
                s = s.astype(str).str.strip().str.lower().map({"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0,
                                                               "success": 1, "fail": 0, "failure": 0})
            if s.isna().any():
                raise ValueError("could not read every value of the 'success' column as 0/1")
            df["success"] = s.astype(int)
            if not df["success"].isin([0, 1]).all():
                raise ValueError("'success' must be 0/1 or true/false")
            self.long = df
            agg = (df.groupby([self.policy, self.task], sort=False)["success"].agg(["sum", "count"])
                   .reset_index().rename(columns={"sum": "successes", "count": "trials"}))
            if self.pair_by and self.pair_by not in df.columns:
                raise ValueError(f"pair_by column {self.pair_by!r} not found")
        else:
            raise ValueError("need either a 'success' column (one row per rollout) or 'successes' and 'trials'")
        if (agg["successes"] > agg["trials"]).any():
            raise ValueError("successes exceed trials somewhere")
        self.agg = agg

    @classmethod
    def from_csv(cls, path: str, **kw) -> "Results":
        return cls(pd.read_csv(path, sep=None, engine="python"), **kw)

    @property
    def policies(self) -> List[str]:
        return list(dict.fromkeys(self.agg[self.policy]))

    @property
    def tasks(self) -> List[str]:
        return list(dict.fromkeys(self.agg[self.task]))

    # ------------------------------------------------------------------ rates
    def rates(self) -> pd.DataFrame:
        rows = []
        for _, r in self.agg.iterrows():
            lo, hi = rate_ci(int(r.successes), int(r.trials), self.conf, self.method)
            rows.append({"policy": r[self.policy], "task": r[self.task], "successes": int(r.successes),
                         "trials": int(r.trials), "rate": r.successes / r.trials, "lo": lo, "hi": hi})
        if len(self.tasks) > 1:
            for p in self.policies:
                sub = self.agg[self.agg[self.policy] == p]
                k, n = int(sub.successes.sum()), int(sub.trials.sum())
                lo, hi = rate_ci(k, n, self.conf, self.method)
                rows.append({"policy": p, "task": ALL, "successes": k, "trials": n, "rate": k / n, "lo": lo, "hi": hi})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ comparisons
    def compare_to(self, baseline: str, test: str = "boschloo", alternative: str = "two-sided") -> pd.DataFrame:
        """Every other policy vs ``baseline``, per task, Holm-corrected across all
        comparisons; plus a Cochran–Mantel–Haenszel test pooled over tasks."""
        if baseline not in self.policies:
            raise ValueError(f"baseline {baseline!r} not among policies {self.policies}")
        rows = []
        for p in self.policies:
            if p == baseline:
                continue
            tables = []
            for t in self.tasks:
                a = self.agg[(self.agg[self.policy] == p) & (self.agg[self.task] == t)]
                b = self.agg[(self.agg[self.policy] == baseline) & (self.agg[self.task] == t)]
                if a.empty or b.empty:
                    continue
                ka, na = int(a.successes.iloc[0]), int(a.trials.iloc[0])
                kb, nb = int(b.successes.iloc[0]), int(b.trials.iloc[0])
                if self.pair_by and self.long is not None:
                    la = self.long[(self.long[self.policy] == p) & (self.long[self.task] == t)]
                    lb = self.long[(self.long[self.policy] == baseline) & (self.long[self.task] == t)]
                    merged = la.merge(lb, on=self.pair_by, suffixes=("_a", "_b"))
                    if len(merged) == 0:
                        raise ValueError(f"no shared {self.pair_by!r} values between {p} and {baseline} on {t}")
                    c = compare_paired(merged["success_a"], merged["success_b"], p, baseline, alternative, self.conf)
                else:
                    c = compare(ka, na, kb, nb, p, baseline, test=test, alternative=alternative, conf=self.conf)
                tables.append(np.array([[ka, na - ka], [kb, nb - kb]]))
                need = None
                if not c.significant and ka / na != kb / nb:
                    try:
                        if c.paired:
                            m = c.n_a
                            pa_only, pb_only = c.discordant[0] / m, c.discordant[1] / m
                            # shrink zero discordant counts a little so planning stays finite
                            pa_only, pb_only = max(pa_only, 0.5 / m), max(pb_only, 0.5 / m)
                            need = n_pairs(pa_only, pb_only, 0.8, 1 - self.conf, alternative, n_max=3000)
                        else:
                            need = n_per_arm(min(max(ka / na, 0.01), 0.99), min(max(kb / nb, 0.01), 0.99),
                                             0.8, 1 - self.conf, alternative, n_max=3000)
                    except ValueError:
                        need = None
                rows.append({"policy": p, "baseline": baseline, "task": t, "diff": c.diff, "lo": c.ci[0], "hi": c.ci[1],
                             "p": c.p_value, "test": c.test, "prob_better": c.prob_a_better, "n_for_80pct_power": need})
            if len(tables) > 1:
                _, p_cmh, mh_or = cmh_test(tables)
                w = [t[0].sum() * t[1].sum() / t.sum() for t in tables]
                d = [t[0, 0] / t[0].sum() - t[1, 0] / t[1].sum() for t in tables]
                rows.append({"policy": p, "baseline": baseline, "task": ALL, "diff": float(np.average(d, weights=w)),
                             "lo": np.nan, "hi": np.nan, "p": p_cmh, "test": "CMH", "prob_better": np.nan,
                             "n_for_80pct_power": None})
        out = pd.DataFrame(rows)
        if len(out):
            out["p_holm"] = holm(out["p"].tolist())
        return out

    # ------------------------------------------------------------------ rendering
    def _cell(self, r) -> str:
        return f"{r.rate:.0%} [{r.lo:.0%}, {r.hi:.0%}] ({r.successes}/{r.trials})"

    def markdown(self, baseline: Optional[str] = None, **kw) -> str:
        rates = self.rates()
        tasks = list(dict.fromkeys(rates["task"]))
        lines = ["| policy | " + " | ".join(tasks) + " |", "|---" * (len(tasks) + 1) + "|"]
        for p in self.policies:
            cells = []
            for t in tasks:
                r = rates[(rates.policy == p) & (rates.task == t)]
                cells.append(self._cell(r.iloc[0]) if len(r) else "—")
            lines.append(f"| {p} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(f"Success rate [{self.conf:.0%} {self.method.title()} CI] (successes/trials).")
        if baseline:
            comp = self.compare_to(baseline, **kw)
            lines += ["", f"Comparison against **{baseline}** (Holm-corrected across {len(comp)} tests):", "",
                      "| policy | task | difference | CI | p | p (Holm) | note |", "|---|---|---|---|---|---|---|"]
            for _, r in comp.iterrows():
                ci = "—" if np.isnan(r.lo) else f"[{r.lo:+.0%}, {r.hi:+.0%}]"
                note = ""
                if r.p_holm >= 1 - self.conf and pd.notna(r.n_for_80pct_power):
                    unit = "pairs" if r.test == "exact McNemar" else "rollouts/policy"
                    note = f"~{int(r.n_for_80pct_power)} {unit} for 80% power"
                lines.append(f"| {r.policy} | {r.task} | {r['diff']:+.0%} | {ci} | {r.p:.3g} | {r.p_holm:.3g} | {note} |")
        return "\n".join(lines)

    def latex(self, digits: int = 0) -> str:
        rates = self.rates()
        tasks = list(dict.fromkeys(rates["task"]))
        f = f"{{:.{digits}f}}"
        lines = [r"\begin{tabular}{l" + "c" * len(tasks) + "}", r"\toprule",
                 "Policy & " + " & ".join(t.replace("_", r"\_") for t in tasks) + r" \\", r"\midrule"]
        for p in self.policies:
            cells = []
            for t in tasks:
                r = rates[(rates.policy == p) & (rates.task == t)]
                if len(r):
                    r = r.iloc[0]
                    cells.append(f"{f.format(100 * r.rate)}\\% {{\\scriptsize[{f.format(100 * r.lo)}, {f.format(100 * r.hi)}]}}")
                else:
                    cells.append("--")
            lines.append(p.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}",
                  f"% success rate with {self.conf:.0%} {self.method.title()} interval; generated by rollout-stats"]
        return "\n".join(lines)
