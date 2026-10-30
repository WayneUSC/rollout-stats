# Changelog

## 0.1.0 — 2026-10-29

First release.

- Success-rate intervals: Wilson, Clopper–Pearson, Jeffreys, Agresti–Coull.
- Comparisons: Boschloo / Fisher / Barnard exact tests, Newcombe intervals, Bayesian P(A > B); paired exact McNemar.
- Cochran–Mantel–Haenszel across tasks; Holm correction across a results table.
- Planning: exact power (Fisher, McNemar), rollouts per policy, rollouts for a target interval width, minimum
  detectable effect.
- Anytime-valid sequential A/B test and confidence sequences.
- Results tables from one-row-per-rollout CSVs: Markdown, LaTeX (booktabs), HTML with forest plots.
- CLI: `ci`, `compare`, `paired`, `plan`, `table`, `sequential`.
