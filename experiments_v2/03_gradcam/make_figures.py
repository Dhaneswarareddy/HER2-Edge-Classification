"""Figures for experiment 3. Example selection is deterministic and rule-based (no hand-picking):
  per class: the most confident CORRECT tile and the most confident WRONG tile (if any);
  2+ focus: 3 most confident correct 2+, every wrong true-2+ (up to 6, most confident first), 3 most confident
            tiles wrongly called 2+;
  background attention: the 6 tiles with the highest bg_ratio (attention on blank glass relative to its area).
Each example has three panels: original input, Grad-CAM heatmap, overlay.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gradcam as G  # noqa: E402

L, OUT = G.L, G.OUT
SOURCES = {
    "BCI_test_448": lambda: G.bci_test(448), "BCI_test_224": lambda: G.bci_test(224),
    "ext_test_raw_448": lambda: G.ext_test(448), "ext_test_raw_224": lambda: G.ext_test(224),
    "ext_test_p99_224": lambda: G.ext_test(224, calibrated=True),
}
SIZE = {"BCI_test_448": 448, "BCI_test_224": 224, "ext_test_raw_448": 448, "ext_test_raw_224": 224, "ext_test_p99_224": 224}


def load_set(tag):
    z = np.load(OUT / f"cams_{tag}.npz", allow_pickle=True)
    st = pd.read_csv(OUT / "cam_stats_per_tile.csv")
    st = st[st.set == tag].reset_index(drop=True)
    assert (st.name.values == z["names"]).all()
    return st, z["cam"], SOURCES[tag]()[2]


def pick_per_class(st):
    out = []
    for c in range(4):
        d = st[st.label == c]
        for ok in (True, False):
            e = d[(d.label == d.pred) == ok].sort_values("confidence", ascending=False)
            if len(e):
                out.append((e.index[0], "correct" if ok else "wrong"))
    return out


def pick_2plus(st):
    ok = st[(st.label == 2) & (st.pred == 2)].sort_values("confidence", ascending=False).index[:3]
    wr = st[(st.label == 2) & (st.pred != 2)].sort_values("confidence", ascending=False).index[:6]
    fp = st[(st.label != 2) & (st.pred == 2)].sort_values("confidence", ascending=False).index[:3]
    return [(i, "correct 2+") for i in ok] + [(i, "missed 2+") for i in wr] + [(i, "false 2+") for i in fp]


def pick_background(st):
    return [(i, "high background attention") for i in st.dropna(subset=["bg_ratio"]).sort_values("bg_ratio", ascending=False).index[:6]]


def panel_figure(tag, picks, title, fname):
    st, cams, load = load_set(tag)
    n = len(picks)
    fig, axes = plt.subplots(n, 3, figsize=(7.2, 2.35 * n))
    axes = np.atleast_2d(axes)
    for r, (i, kind) in enumerate(picks):
        img = np.clip(load(st.name[i]), 0, 255).astype(np.uint8)  # display only
        cam = G.upsample(cams[i], SIZE[tag])
        cam = cam / cam.max() if cam.max() > 0 else cam
        axes[r, 0].imshow(img); axes[r, 1].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[r, 2].imshow(img); axes[r, 2].imshow(cam, cmap="jet", alpha=0.4, vmin=0, vmax=1)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
        s = st.loc[i]
        axes[r, 0].set_ylabel(f"{kind}\ntrue {L.LEVELS[s.label]} / pred {L.LEVELS[s.pred]}\np={s.confidence:.2f}", fontsize=7.5)
        axes[r, 2].set_title(f"DAB ratio {s.dab_ratio:.2f} | bg ratio {s.bg_ratio:.2f}" if s.bg_ratio == s.bg_ratio
                             else f"DAB ratio {s.dab_ratio:.2f} | <5% background", fontsize=7)
        if r == 0:
            axes[r, 0].set_title("Input", fontsize=8.5); axes[r, 1].set_title("Grad-CAM (max-normalised)", fontsize=8.5)
    fig.suptitle(title, fontsize=9.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(L.guard(OUT / fname), dpi=300)
    plt.close(fig)
    return [dict(figure=fname, set=tag, name=st.name[i], kind=k, label=int(st.label[i]), pred=int(st.pred[i]),
                 confidence=float(st.confidence[i]), dab_ratio=float(st.dab_ratio[i]), bg_ratio=float(st.bg_ratio[i]))
            for i, k in picks]


def summary_figure():
    st = pd.read_csv(OUT / "cam_stats_per_tile.csv")
    st["correct"] = st.label == st.pred
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    tags = list(SIZE)
    for ax, col, lab in zip(axes, ["dab_ratio", "bg_ratio", "top_dab_mass"],
                            ["DAB attention ratio (>1 = on brown)", "Background attention ratio (<1 = avoids glass)",
                             "CAM mass on top-10% DAB pixels (0.10 = uniform)"]):
        data, names = [], []
        for t in tags:
            for ok in (True, False):
                v = st[(st.set == t) & (st.correct == ok)][col].dropna()
                data.append(v.values); names.append(f"{t.replace('_test', '')}\n{'ok' if ok else 'wrong'} n={len(v)}")
        ax.boxplot(data, tick_labels=names, showfliers=False)
        ax.tick_params(axis="x", labelsize=5.5, rotation=60)
        ax.set_title(lab, fontsize=9)
        if col == "dab_ratio":
            ax.axhline(1, ls=":", c="k")
        if col == "bg_ratio":
            ax.axhline(1, ls=":", c="k")
        if col == "top_dab_mass":
            ax.axhline(0.1, ls=":", c="k")
    fig.tight_layout()
    fig.savefig(L.guard(OUT / "fig_cam_statistics.png"), dpi=300)
    plt.close(fig)


def main():
    rows = []
    rows += panel_figure("BCI_test_448", pick_per_class(pd.read_csv(OUT / "cam_stats_per_tile.csv").query("set=='BCI_test_448'").reset_index(drop=True)),
                         "Grad-CAM, EfficientNetV2B0 448 px, BCI test: most confident correct and wrong tile per class", "fig_gradcam_bci448_per_class.png")
    rows += panel_figure("BCI_test_224", pick_per_class(pd.read_csv(OUT / "cam_stats_per_tile.csv").query("set=='BCI_test_224'").reset_index(drop=True)),
                         "Grad-CAM, EfficientNetV2B0 224 px, BCI test: most confident correct and wrong tile per class", "fig_gradcam_bci224_per_class.png")
    for tag, nm in [("BCI_test_448", "bci448"), ("ext_test_p99_224", "ext224_p99")]:
        st = pd.read_csv(OUT / "cam_stats_per_tile.csv").query(f"set=='{tag}'").reset_index(drop=True)
        rows += panel_figure(tag, pick_2plus(st), f"Grad-CAM 2+ focus ({tag}): correct 2+, missed 2+, false 2+", f"fig_gradcam_{nm}_2plus.png")
    for tag, nm in [("ext_test_raw_224", "ext224_raw"), ("ext_test_p99_224", "ext224_p99"), ("ext_test_raw_448", "ext448_raw")]:
        st = pd.read_csv(OUT / "cam_stats_per_tile.csv").query(f"set=='{tag}'").reset_index(drop=True)
        rows += panel_figure(tag, pick_per_class(st), f"Grad-CAM ({tag}), external test (exploratory): most confident correct and wrong tile per class",
                             f"fig_gradcam_{nm}_per_class.png")
    st = pd.read_csv(OUT / "cam_stats_per_tile.csv")
    for tag, nm in [("BCI_test_448", "bci448"), ("ext_test_raw_448", "ext448_raw")]:
        s = st.query(f"set=='{tag}'").reset_index(drop=True)
        rows += panel_figure(tag, pick_background(s), f"Highest background attention ({tag})", f"fig_gradcam_{nm}_background.png")
    summary_figure()
    L.write_csv(OUT / "figure_examples.csv", pd.DataFrame(rows))


if __name__ == "__main__":
    main()
