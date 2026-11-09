"""Step 4 (FYP3): choose the loss per model and the best model per tier, FP32 only.

Same principles as FYP2/src/select_models.py, adapted because FYP3 stops before quantisation:
  * every decision uses the VALIDATION split; test numbers are reported only;
  * loss (CE vs CORN) is chosen only for models trained with both: higher val QWK (4 d.p.), ties ->
    higher val macro-F1 (FYP2 used full-INT8 robustness here, which does not exist in FYP3);
  * tier winner among BASELINE runs (the FYP2 pipeline reproduced): best val QWK; runs within
    QWK_TOL of it are equivalent and the one with fewer parameters wins (FYP2 used Mac latency of the
    quantised model, which is out of scope here).
  * improvement variants are reported separately ("best with improvements"), chosen by the same rule.

Output: results/selection.json
"""
import json

import pandas as pd

import config as C

QWK_TOL = 0.005


def load_runs():
    rows = []
    for p in sorted(C.RUNS_DIR.glob("*/summary.json")):
        s = json.load(open(p))
        rows.append(dict(run=p.parent.name, model=s["model"], loss=s["loss"], variant=s.get("variant") or "baseline",
                         tier=s["tier"], params=s["params"], best_epoch=s["best_epoch"], epochs_run=s["epochs_run"],
                         train_min=s["train_time_s"] / 60, image_size=s.get("image_size", C.IMAGE_SIZE),
                         **{f"val_{k}": v for k, v in s["val"].items() if k != "confusion_matrix"},
                         **{f"test_{k}": v for k, v in s["test"].items() if k != "confusion_matrix"}))
    return pd.DataFrame(rows)


def choose_losses(runs):
    base = runs[runs.variant == "baseline"].copy()
    base["qwk_r"], base["f1_r"] = base.val_qwk.round(4), base.val_macro_f1.round(4)
    out = {}
    for model, g in base.groupby("model"):
        out[model] = g.sort_values(["qwk_r", "f1_r"], ascending=False).iloc[0]["loss"]
    return out


def pick(cands):
    top = cands.val_qwk.max()
    eq = cands[cands.val_qwk >= top - QWK_TOL].sort_values(["params", "val_qwk"], ascending=[True, False])
    w = eq.iloc[0]
    return dict(run=w.run, model=w.model, loss=w.loss, variant=w.variant, params=int(w.params),
                val_qwk=float(w.val_qwk), val_accuracy=float(w.val_accuracy), val_macro_f1=float(w.val_macro_f1),
                test_qwk=float(w.test_qwk), test_accuracy=float(w.test_accuracy),
                test_macro_f1=float(w.test_macro_f1), equivalent_candidates=list(eq.run))


def main():
    runs = load_runs()
    losses = choose_losses(runs)
    base = runs[(runs.variant == "baseline") & runs.apply(lambda r: losses.get(r.model) == r.loss, axis=1)]
    winners = {t: pick(base[base.tier == t]) for t in ["heavy", "mid", "lite"] if (base.tier == t).any()}
    improved = {t: pick(runs[runs.tier == t]) for t in ["heavy", "mid", "lite"] if (runs.tier == t).any()}
    both = runs[runs.variant == "baseline"].pivot_table(index="model", columns="loss", values="val_qwk")
    both = both.dropna() if {"ce", "corn"} <= set(both.columns) else both.iloc[0:0]
    ce_vs_corn = dict(models_compared=int(len(both)),
                      corn_better=int((both["corn"] > both["ce"] + 1e-4).sum()) if len(both) else 0,
                      ce_better=int((both["ce"] > both["corn"] + 1e-4).sum()) if len(both) else 0)
    sel = dict(rule=__doc__, chosen_loss_per_model=losses, ce_vs_corn=ce_vs_corn,
               tier_winners_baseline=winners, tier_best_with_improvements=improved)
    json.dump(sel, open(C.RESULTS_DIR / f"selection{C.SUFFIX}.json", "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in sel.items() if k != "rule"}, indent=2, default=float))


if __name__ == "__main__":
    main()
