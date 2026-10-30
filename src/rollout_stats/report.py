"""Self-contained HTML report with forest plots."""

from __future__ import annotations

import html
from typing import List, Optional

import numpy as np

_CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e4e3de;--accent:#3b5bdb;--base:#868e96;
--good:#2b8a3e;--bad:#d6336c;--goodbg:#ebf7ee;--badbg:#fdecf2;--nbg:#f1f1ef}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141413;--card:#1e1e1c;--ink:#ecebe6;--muted:#a3a29b;
--line:#33332f;--accent:#7b93f0;--base:#8b8b84;--good:#58c27a;--bad:#f06595;--goodbg:#1d2e22;--badbg:#351c27;--nbg:#262624}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:960px;margin:0 auto;padding:32px 16px 64px}.tool{font-size:13px;color:var(--muted);letter-spacing:.04em;text-transform:uppercase}
h1{font-size:28px;margin:4px 0 8px;font-weight:650}.facts{color:var(--muted);font-size:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px 22px;margin:16px 0;overflow-x:auto}
.card h2{font-size:18px;margin:0 0 4px}.q{color:var(--muted);margin:0 0 10px;font-size:14px}
table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
th{text-align:left;color:var(--muted);font-weight:600;font-size:13px;padding:6px}td{padding:7px 6px;border-top:1px solid var(--line)}
.pill{font-size:12px;padding:2px 10px;border-radius:999px;font-weight:700;white-space:nowrap}
.pill.good{background:var(--goodbg);color:var(--good)}.pill.bad{background:var(--badbg);color:var(--bad)}.pill.n{background:var(--nbg);color:var(--muted)}
svg{max-width:100%;height:auto}footer{color:var(--muted);font-size:13px;margin-top:32px}
"""


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _forest(rows, baseline: Optional[str], width: int = 860) -> str:
    """rows: list of (label, rate, lo, hi)."""
    lw, rp, top, rh = 170, 140, 10, 30
    h = top + rh * len(rows) + 30

    def X(v):
        return lw + v * (width - lw - rp)

    out = [f'<svg viewBox="0 0 {width} {h}" xmlns="http://www.w3.org/2000/svg" font-size="13" role="img" aria-label="success rates">']
    for k in range(6):
        v = k / 5
        out.append(f'<line x1="{X(v):.1f}" y1="{top}" x2="{X(v):.1f}" y2="{h - 24}" stroke="var(--line)"/>')
        out.append(f'<text x="{X(v):.1f}" y="{h - 8}" fill="var(--muted)" text-anchor="middle">{v:.0%}</text>')
    for i, (label, p, lo, hi) in enumerate(rows):
        y = top + i * rh + rh / 2
        color = "var(--base)" if label == baseline else "var(--accent)"
        out.append(f'<text x="{lw - 10}" y="{y + 4:.1f}" fill="var(--ink)" text-anchor="end">{_e(label)}</text>')
        out.append(f'<line x1="{X(lo):.1f}" y1="{y:.1f}" x2="{X(hi):.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        out.append(f'<circle cx="{X(p):.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
        out.append(f'<text x="{width - rp + 10}" y="{y + 4:.1f}" fill="var(--muted)">{p:.0%} [{lo:.0%}, {hi:.0%}]</text>')
    out.append("</svg>")
    return "".join(out)


def render_html(results, baseline: Optional[str] = None, title: str = "Policy evaluation", **compare_kw) -> str:
    from . import __version__

    rates = results.rates()
    tasks = list(dict.fromkeys(rates["task"]))
    parts: List[str] = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body><main>",
        '<div class="tool">rollout-stats · is the improvement real?</div>',
        f"<h1>{_e(title)}</h1>",
        f'<div class="facts">{len(results.policies)} policies · {len(results.tasks)} task(s) · '
        f'{int(rates[rates.task != "all tasks"].trials.sum()) if "all tasks" in tasks else int(rates.trials.sum())} rollouts · '
        f'{results.conf:.0%} {results.method.title()} intervals</div>',
    ]
    comp = results.compare_to(baseline, **compare_kw) if baseline else None
    for t in tasks:
        sub = rates[rates.task == t]
        rows = [(r.policy, r.rate, r.lo, r.hi) for r in sub.itertuples()]
        parts.append(f'<section class="card"><h2>{_e(t)}</h2>')
        parts.append(f'<p class="q">Success rate with {results.conf:.0%} interval. Overlapping intervals do not by themselves '
                     "mean “no difference”; see the comparison below.</p>")
        parts.append(_forest(rows, baseline))
        if comp is not None and len(comp):
            c = comp[comp.task == t]
            if len(c):
                parts.append('<table><tr><th>vs ' + _e(baseline) + '</th><th>difference</th><th>CI</th><th>p</th>'
                             '<th>p (Holm)</th><th>verdict</th></tr>')
                for r in c.itertuples():
                    alpha = 1 - results.conf
                    if r.p_holm < alpha:
                        pill = '<span class="pill good">better</span>' if r.diff > 0 else '<span class="pill bad">worse</span>'
                    else:
                        unit = "pairs" if r.test == "exact McNemar" else "rollouts/policy"
                        need = (f" — ~{int(r.n_for_80pct_power)} {unit} for 80% power"
                                if r.n_for_80pct_power == r.n_for_80pct_power and r.n_for_80pct_power else "")
                        pill = f'<span class="pill n">not significant</span><span class="q">{_e(need)}</span>'
                    ci = "—" if np.isnan(r.lo) else f"[{r.lo:+.0%}, {r.hi:+.0%}]"
                    parts.append(f"<tr><td>{_e(r.policy)}</td><td>{r.diff:+.0%}</td><td>{ci}</td><td>{r.p:.3g}"
                                 f"<span class=\"q\"> {_e(r.test)}</span></td><td>{r.p_holm:.3g}</td><td>{pill}</td></tr>")
                parts.append("</table>")
        parts.append("</section>")
    parts.append(f"<footer>rollout-stats {__version__}. Exact tests (Boschloo / McNemar), Newcombe intervals, "
                 "Holm correction across all comparisons, Cochran–Mantel–Haenszel across tasks.</footer></main></body></html>")
    return "".join(parts)
