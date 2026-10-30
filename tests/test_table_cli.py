import numpy as np
import pandas as pd
import pytest

import rollout_stats as rs
from rollout_stats.cli import main


@pytest.fixture()
def long_csv(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    rates = {"ours": {"pick": 0.85, "pour": 0.6}, "baseline": {"pick": 0.55, "pour": 0.5}}
    for task in ("pick", "pour"):
        difficulty = rng.random(30)  # shared per-seed difficulty -> correlated outcomes
        for policy, r in rates.items():
            for seed in range(30):
                rows.append({"policy": policy, "task": task, "seed": seed,
                             "success": int(difficulty[seed] < r[task])})
    path = tmp_path / "rollouts.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_results_long_and_aggregated(long_csv):
    res = rs.Results.from_csv(str(long_csv))
    rates = res.rates()
    assert set(rates.task) == {"pick", "pour", "all tasks"}
    comp = res.compare_to("baseline")
    assert {"pick", "pour", "all tasks"} == set(comp.task)
    assert (comp.p_holm >= comp.p - 1e-12).all()
    agg = res.agg.rename(columns={})
    res2 = rs.Results(agg)
    assert res2.rates()["successes"].sum() == rates["successes"].sum()


def test_paired_table_is_more_powerful(long_csv):
    unpaired = rs.Results.from_csv(str(long_csv)).compare_to("baseline")
    paired = rs.Results.from_csv(str(long_csv), pair_by="seed").compare_to("baseline")
    pick_u = unpaired[unpaired.task == "pick"].p.iloc[0]
    pick_p = paired[paired.task == "pick"].p.iloc[0]
    assert paired[paired.task == "pick"].test.iloc[0] == "exact McNemar"
    assert pick_p < pick_u


def test_renderers(long_csv):
    res = rs.Results.from_csv(str(long_csv))
    md = res.markdown("baseline")
    assert "| ours |" in md and "Holm" in md
    tex = res.latex()
    assert r"\toprule" in tex and r"\%" in tex
    from rollout_stats.report import render_html

    html = render_html(res, "baseline")
    assert html.count("<svg") == 3


def test_bad_inputs(tmp_path):
    with pytest.raises(ValueError):
        rs.Results(pd.DataFrame({"policy": ["a"], "x": [1]}))
    with pytest.raises(ValueError):
        rs.Results(pd.DataFrame({"policy": ["a"], "successes": [5], "trials": [3]}))
    with pytest.raises(ValueError):
        rs.rate_ci(5, 3)


def test_cli(long_csv, tmp_path, capsys):
    assert main(["ci", "17", "25"]) == 0
    assert "48.4%" in capsys.readouterr().out
    assert main(["compare", "17/25", "11/25", "--names", "ours,base"]) == 0
    assert "inconclusive" in capsys.readouterr().out
    assert main(["plan", "--p-a", "0.8", "--p-b", "0.6", "--sims", "100"]) == 0
    assert "rollouts per policy" in capsys.readouterr().out
    assert main(["plan", "--p", "0.7", "--halfwidth", "0.1"]) == 0
    assert main(["plan", "--p-b", "0.5", "--n", "25"]) == 0
    html, tex = tmp_path / "r.html", tmp_path / "t.tex"
    assert main(["table", str(long_csv), "--baseline", "baseline", "--pair-by", "seed", "--html", str(html),
                 "--latex", str(tex)]) == 0
    assert html.exists() and tex.exists()
    pairs = tmp_path / "pairs.csv"
    pd.DataFrame({"ours": [1] * 25, "base": [0, 1] * 12 + [0]}).to_csv(pairs, index=False)
    assert main(["sequential", str(pairs), "--a", "ours", "--b", "base"]) == 0
    assert "DECISION" in capsys.readouterr().out
    assert main(["paired", str(pairs), "--a", "ours", "--b", "base"]) == 0
    assert main(["plan"]) == 2
    assert main(["compare", "17/25", "11/25", "--test", "fisher"]) == 0
