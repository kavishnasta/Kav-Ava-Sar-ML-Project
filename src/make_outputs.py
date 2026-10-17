"""Phase 6: figures (vector PDF in plots/), tables (CSV in results/tables/), top confusions, results/summary.md.
Computed only from results/grid_*.csv, baseline_*.csv, ablation_*.csv and preds_*.npz (SVM only)."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import config
import corruptions
import datasets

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE = config.MODEL_COLORS["SVM"]
FAMS = list(corruptions.FAMILIES)
pretty = lambda n: n.replace("_", " ")


def rp(name, ds, ext="csv"):
    return os.path.join(config.RESULTS_DIR, f"{name}_{ds}.{ext}")


def save(fig, name):
    fig.savefig(os.path.join(config.PLOTS_DIR, name), format="pdf", bbox_inches="tight")
    plt.close(fig)


class Data:
    def __init__(self, ds):
        self.ds, self.k = ds, datasets.N_CLASSES[ds]
        self.grid = pd.read_csv(rp("grid", ds))
        self.base = pd.read_csv(rp("baseline", ds))
        self.preds = dict(np.load(rp("preds", ds, "npz")))
        self.corrs = [c for c in corruptions.NAMES if c in set(self.grid.corruption)]

    def acc(self, corr, sev):
        r = self.grid[(self.grid.corruption == corr) & (self.grid.severity == sev)]
        return float(r.accuracy.iloc[0])

    def base_val(self, split, col="accuracy"):
        return float(self.base[self.base.split == split][col].iloc[0])

    @property
    def clean_ref(self):                           # clean accuracy on the same samples as the corruption grid
        return self.base_val("test_subset" if (self.base.split == "test_subset").any() else "test")

    @property
    def sev5(self):
        return pd.Series({c: self.acc(c, 5) for c in self.corrs})


def fig_confusion(D):
    worst = D.sev5.idxmin()
    panels = [("clean", D.preds["y_true_clean"], D.preds["clean__0"]),
              (f"worst corruption, severity 5: {pretty(worst)}", D.preds["y_true_sub"], D.preds[f"{worst}__5"])]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
    for ax, (title, yt, yp) in zip(axes, panels):
        cm = confusion_matrix(yt, yp, labels=range(D.k)).astype(float)
        rs = cm.sum(1, keepdims=True)
        im = ax.pcolormesh(np.divide(cm, rs, out=np.full_like(cm, np.nan), where=rs > 0), cmap="Blues", vmin=0, vmax=1)
        ax.set_ylim(D.k, 0)
        ax.set_aspect("equal")
        t = np.arange(0, D.k, 5)
        ax.set_xticks(t + 0.5, t); ax.set_yticks(t + 0.5, t)
        ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
        ax.set_title(f"{title}\naccuracy {np.mean(yt == yp):.3f}", fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="row-normalised share")
    fig.suptitle(f"{D.ds.upper()} - SVM confusion matrices", fontsize=13)
    save(fig, f"fig_confusion_{D.ds}.pdf")


def fig_heatmap(D):
    A = np.array([[D.acc(c, s) for s in range(1, 6)] for c in D.corrs])
    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(A, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(A):
        ax.text(j, i, f"{v * 100:.1f}", ha="center", va="center", fontsize=9.5, color="white" if v < 0.6 else "black")
    ax.set_xticks(range(5), range(1, 6))
    ax.set_yticks(range(len(D.corrs)), [pretty(c) for c in D.corrs])
    ax.set_xlabel("severity")
    ax.set_title(f"{D.ds.upper()} - SVM accuracy (%)\nclean test accuracy {D.base_val('test') * 100:.1f}%")
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="accuracy")
    save(fig, f"fig_heatmap_{D.ds}.pdf")


def fig_lines(D):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
    for ax, fam in zip(axes, FAMS):
        for c, m in zip([c for c in corruptions.FAMILIES[fam] if c in D.corrs], "osv^"):
            ax.plot(range(6), [D.clean_ref] + [D.acc(c, s) for s in range(1, 6)], marker=m, ms=4, lw=1.8, label=pretty(c))
        ax.set_title(fam); ax.set_xlabel("severity (0 = clean)"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("SVM accuracy")
    fig.suptitle(f"{D.ds.upper()} - accuracy vs severity", y=1.02, fontsize=13)
    save(fig, f"fig_severity_lines_{D.ds}.pdf")


def fig_feature_ablation(data):
    names = [d for d in data if os.path.exists(rp("ablation_features", d))]
    fig, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 4), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], names):
        A, G = pd.read_csv(rp("ablation_features", d)), data[d].grid
        def val(g, sev):
            sub = G[G.severity == sev] if g == "all" else A[(A.model == f"SVM[{g}]") & (A.severity == sev)]
            return sub.accuracy.mean() if len(sub) else np.nan
        gs = [g for g in ["HOG", "HSV", "Hu", "HOG+HSV", "all"] if not np.isnan(val(g, 0))]
        for i, (sev, lab, a) in enumerate(((0, "clean", 1.0), (3, "severity 3 (mean)", 0.65), (5, "severity 5 (mean)", 0.35))):
            ax.bar(np.arange(len(gs)) + (i - 1) * 0.27, [val(g, sev) for g in gs], 0.27, color=BLUE, alpha=a, label=lab)
        ax.set_xticks(range(len(gs)), gs); ax.set_title(d.upper()); ax.grid(axis="y", alpha=0.25)
    axes[0][0].set_ylabel("SVM accuracy"); axes[0][0].legend(frameon=False, fontsize=9)
    fig.suptitle("Feature-group ablation", y=1.03, fontsize=13)
    save(fig, "fig_ablation_features.pdf")


def fig_preproc_ablation():
    path = rp("ablation_preproc", "gtsrb")
    if not os.path.exists(path):
        return
    A = pd.read_csv(path)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
    for ax, fam in zip(axes, FAMS):
        for v, ls in zip(dict.fromkeys(A.model), ["-", "--", ":"]):
            clean = A[(A.model == v) & (A.severity == 0)].accuracy.iloc[0]
            ys = [clean] + [A[(A.model == v) & (A.severity == s) & (A.family == fam)].accuracy.mean() for s in (1, 3, 5)]
            ax.plot([0, 1, 3, 5], ys, ls, color=BLUE, lw=2.2, marker="o", label=v)
        ax.set_title(fam); ax.set_xlabel("severity"); ax.set_xticks([0, 1, 3, 5]); ax.set_ylim(0, 1.02); ax.grid(alpha=0.25)
    axes[0].set_ylabel("SVM accuracy (family mean)"); axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("GTSRB preprocessing ablation: median blur / CLAHE on vs off", y=1.02, fontsize=13)
    save(fig, "fig_ablation_preproc_gtsrb.pdf")


def top_confusions(D):
    worst = D.sev5.idxmin()
    rows = []
    for tag, yt, yp in (("clean", D.preds["y_true_clean"], D.preds["clean__0"]),
                        (f"{worst}_sev5", D.preds["y_true_sub"], D.preds[f"{worst}__5"])):
        cm = confusion_matrix(yt, yp, labels=range(D.k))
        off = cm.copy(); np.fill_diagonal(off, 0)
        for flat in np.argsort(off, axis=None)[::-1][:10]:
            t, p = divmod(int(flat), D.k)
            rows.append(dict(condition=tag, true_class=t, predicted_class=p, count=int(off[t, p]), share_of_true_class=off[t, p] / max(1, cm[t].sum())))
    pd.DataFrame(rows).to_csv(os.path.join(config.RESULTS_DIR, f"top_confusions_{D.ds}_SVM.csv"), index=False)


def build_tables(data):
    head, fam, bw = [], [], []
    for ds, D in data.items():
        s5, ref = D.sev5, D.clean_ref
        head.append(dict(dataset=ds, clean_acc=D.base_val("test"), clean_macro_f1=D.base_val("test", "macro_f1"),
                         mean_acc_sev5=s5.mean(), retention_sev5=s5.mean() / ref))
        for f in FAMS:
            cs = [c for c in corruptions.FAMILIES[f] if c in D.corrs]
            m15 = np.mean([[D.acc(c, s) for s in range(1, 6)] for c in cs])
            fam.append(dict(dataset=ds, family=f, mean_acc_sev5=s5[cs].mean(), retention_sev5=s5[cs].mean() / ref,
                            mean_acc_sev1to5=m15, retention_sev1to5=m15 / ref))
        bw.append(dict(dataset=ds, best_corruption=s5.idxmax(), best_acc_sev5=s5.max(), worst_corruption=s5.idxmin(), worst_acc_sev5=s5.min()))
        D.grid.pivot_table(index="corruption", columns="severity", values="accuracy").reset_index().to_csv(
            os.path.join(config.TABLES_DIR, f"accuracy_by_severity_{ds}.csv"), index=False)
    T = dict(headline=pd.DataFrame(head), family_retention=pd.DataFrame(fam), best_worst=pd.DataFrame(bw))
    for k, v in T.items():
        v.to_csv(os.path.join(config.TABLES_DIR, k + ".csv"), index=False)
    return T


def md(df):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_summary(data, T):
    L = ["# Results summary (auto-generated by src/make_outputs.py)", ""]
    for ds, D in data.items():
        g = D.grid
        L.append(f"- **{ds.upper()}**: mode `{g['mode'].iloc[0]}`, {int(g[g.severity > 0].n_samples.iloc[0])} test images per corrupted condition, "
                 f"{int(g[g.severity == 0].n_samples.iloc[0])} for clean")
    L += ["", "## Headline (SVM)", "", "clean_* = full clean test set; mean_acc_sev5 = mean over the 14 corruptions at severity 5; "
          "retention = mean_acc_sev5 / clean accuracy on the same samples.", "", md(T["headline"]), "",
          "## Retention per corruption family", "", md(T["family_retention"]), "",
          "## Best and worst corruption (accuracy at severity 5)", "", md(T["best_worst"]), ""]
    for ds in data:
        n = os.path.join(config.RESULTS_DIR, f"dataset_notes_{ds}.txt")
        if os.path.exists(n):
            L += [f"## {ds.upper()} dataset notes", "", "```", open(n).read().strip(), "```", ""]
    open(os.path.join(config.RESULTS_DIR, "summary.md"), "w").write("\n".join(L))


def main():
    args = datasets.add_dataset_arg(argparse.ArgumentParser()).parse_args()
    data = {ds: Data(ds) for ds in (config.DATASETS if args.dataset == "all" else [args.dataset])
            if os.path.exists(rp("grid", ds)) and os.path.exists(rp("preds", ds, "npz"))}
    for D in data.values():
        fig_confusion(D); fig_heatmap(D); fig_lines(D); top_confusions(D)
    fig_feature_ablation(data)
    fig_preproc_ablation()
    write_summary(data, build_tables(data))
    print("outputs written to plots/, results/tables/, results/summary.md")


if __name__ == "__main__":
    main()
