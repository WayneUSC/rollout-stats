"""Generate examples/rollouts.csv: 3 policies x 3 tasks x 20 shared seeds (one row per rollout).

Each seed has a difficulty shared by all policies (same initial object pose), so outcomes
are correlated across policies — exactly the situation where paired tests help.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
rates = {
    "ours": {"pick_cube": 0.85, "open_drawer": 0.70, "pour_water": 0.55},
    "diffusion_policy": {"pick_cube": 0.75, "open_drawer": 0.60, "pour_water": 0.30},
    "act": {"pick_cube": 0.60, "open_drawer": 0.45, "pour_water": 0.25},
}
rows = []
for task in ["pick_cube", "open_drawer", "pour_water"]:
    difficulty = rng.random(20)
    for policy, r in rates.items():
        noise = rng.normal(0, 0.15, 20)
        for seed in range(20):
            rows.append({"policy": policy, "task": task, "seed": seed,
                         "success": int(difficulty[seed] + noise[seed] < r[task])})
pd.DataFrame(rows).to_csv("examples/rollouts.csv", index=False)
print(pd.DataFrame(rows).groupby(["task", "policy"]).success.mean().unstack().round(2))
