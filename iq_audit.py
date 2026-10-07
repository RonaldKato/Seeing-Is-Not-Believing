"""
IQ-Audit: Information Quality Audit of an Open Medical Imaging Repository
=========================================================================
Reproduces every table and figure in the paper
"Seeing Is Not Believing: An Information Quality Audit of Open Medical Imaging Repositories".

Usage
-----
    git clone https://github.com/ieee8023/covid-chestxray-dataset.git ds
    pip install -r requirements.txt
    python iq_audit.py --data ds --out results

Outputs (in --out)
------------------
    table1_dataset_profile.csv      table3_label_association.csv
    table4_shortcut_probes.csv      table5_split_inflation.csv
    completeness.csv                duplicates.csv
    fig2_samples.png ... fig8_split.png
"""
import argparse, os, json, warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C1, C0 = "#C0392B", "#2E86C1"          # COVID / non-COVID colours
SIZE, BORDER = 64, 10                  # resize to 64x64; 10-px border frame
FRONTAL = ["PA", "AP", "AP Supine", "AP Erect"]
COVID = "Pneumonia/Viral/COVID-19"
N_SPLITS, N_REPEATS = 5, 5


# ---------------------------------------------------------------- 1. INGEST
def ingest(data_dir):
    """Load metadata, link images, extract file properties, pixel stats and 64x64 pixels."""
    m = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    x = m[(m.modality == "X-ray") & (m.folder == "images")].copy()
    x["source"] = x.url.fillna("").map(lambda u: urlparse(u).netloc.replace("www.", "") or "unknown")
    rows, pixels = [], []
    for fn in x.filename:
        p = os.path.join(data_dir, "images", fn)
        im = Image.open(p)
        grey = im.convert("L")
        g = np.asarray(grey, np.float32) / 255
        hist = np.histogram(g, bins=256, range=(0, 1))[0]
        pr = hist / hist.sum()
        rows.append(dict(
            w=im.width, h=im.height, mode=im.mode,
            fmt=(im.format or os.path.splitext(fn)[1][1:]).upper(),
            kb=os.path.getsize(p) / 1024,
            mean=g.mean(), std=g.std(),
            entropy=-(pr[pr > 0] * np.log2(pr[pr > 0])).sum(),
            sat=(g >= 0.99).mean() + (g <= 0.01).mean(),
        ))
        pixels.append(np.asarray(grey.resize((SIZE, SIZE), Image.BILINEAR), np.float32).ravel() / 255)
    x = pd.concat([x.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return x, np.stack(pixels)


# --------------------------------------------------------------- 2. PROFILE
COMPLETENESS_FIELDS = {
    "leukocyte_count": "Leukocyte count", "lymphocyte_count": "Lymphocyte count",
    "temperature": "Temperature", "needed_supplemental_O2": "Supplemental O2",
    "pO2_saturation": "pO2 saturation", "intubated": "Intubated", "survival": "Survival",
    "doi": "DOI", "went_icu": "Went to ICU", "RT_PCR_positive": "RT-PCR result",
    "date": "Date", "license": "License", "age": "Age", "clinical_notes": "Clinical notes",
    "finding": "Finding (label)", "sex": "Sex", "location": "Location", "view": "View",
    "url": "Source URL",
}


def completeness(x):
    c = x[list(COMPLETENESS_FIELDS)].replace({"finding": {"todo": None}}).notna().mean() * 100
    return c.rename(index=COMPLETENESS_FIELDS).sort_values().rename("pct_complete")


def cramers_v(ct):
    chi2, p, _, _ = chi2_contingency(ct)
    return np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))), p


def source_groups(fr, k=6):
    top = fr.source.value_counts().index[:k]
    return fr.source.where(fr.source.isin(top), f"other ({fr.source.nunique() - k} sites)")


# ----------------------------------------------------------------- 3. PROBE
def border_mask():
    mb = np.zeros((SIZE, SIZE), bool)
    mb[:BORDER] = mb[-BORDER:] = mb[:, :BORDER] = mb[:, -BORDER:] = True
    return mb


def pixel_model():
    return make_pipeline(StandardScaler(), PCA(50, random_state=0),
                         LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))


def tabular_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))


def splitter(kind, seed):
    if kind == "patient":
        return StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed)
    return StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)


def cross_validate(F, y, groups, make_model, split):
    aucs, baccs = [], []
    for seed in range(N_REPEATS):
        for tr, te in splitter(split, seed).split(F, y, groups):
            clf = make_model().fit(F[tr], y[tr])
            p = clf.predict_proba(F[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
            baccs.append(balanced_accuracy_score(y[te], p > 0.5))
    return np.mean(aucs), np.std(aucs), np.mean(baccs), np.std(baccs)


# ------------------------------------------------------------------- FIGURES
def fig_samples(fr, data_dir, out):
    srcs = ["radiopaedia.org", "eurorad.org", "sciencedirect.com", "pubs.rsna.org"]
    fig, ax = plt.subplots(2, 4, figsize=(11, 6))
    for r, lab in enumerate([1, 0]):
        for c, s in enumerate(srcs):
            d = fr[(fr.source == s) & (fr.y == lab) & (fr.view == "PA")]
            if len(d) == 0:
                d = fr[(fr.source == s) & (fr.y == lab)]
            row = d.sample(1, random_state=3).iloc[0]
            a = ax[r, c]
            a.imshow(Image.open(os.path.join(data_dir, "images", row.filename)).convert("L"), cmap="gray")
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            a.set_title(f"{s}\n{row.w}x{row.h} {row.fmt}, {row.view}", fontsize=8.5)
        ax[r, 0].set_ylabel("COVID-19" if lab else "Non-COVID\npneumonia/other",
                            fontsize=11, color=C1 if lab else C0)
    plt.suptitle("Sample frontal chest X-rays by label and source website", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig2_samples.png"), dpi=160); plt.close()


def fig_completeness(c, out):
    fig, a = plt.subplots(figsize=(8, 6))
    col = ["#C0392B" if v < 50 else ("#E67E22" if v < 90 else "#27AE60") for v in c]
    a.barh(c.index, c.values, color=col)
    a.axvline(90, ls="--", c="gray", lw=1); a.text(90.5, 0.2, "90% threshold", fontsize=8, color="gray")
    for i, v in enumerate(c.values):
        a.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
    a.set_xlim(0, 112)
    a.set_xlabel("Records with a non-missing value (%)")
    a.set_title("Metadata completeness by field", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig3_completeness.png"), dpi=160); plt.close()


def fig_source_label(fr, V, out):
    ct = pd.crosstab(source_groups(fr), fr.y)
    ct["n"] = ct.sum(axis=1); ct["p"] = ct[1] / ct.n; ct = ct.sort_values("p")
    fig, a = plt.subplots(figsize=(8, 4.2)); yy = np.arange(len(ct))
    a.barh(yy, ct.p * 100, color=C1, label="COVID-19")
    a.barh(yy, (1 - ct.p) * 100, left=ct.p * 100, color=C0, label="Non-COVID")
    a.set_yticks(yy, [f"{i}  (n={n})" for i, n in zip(ct.index, ct.n)])
    a.axvline(fr.y.mean() * 100, c="k", ls=":", lw=1)
    a.text(fr.y.mean() * 100 + 1, len(ct) - 0.45, f"overall {fr.y.mean() * 100:.0f}%", fontsize=8)
    a.set_xlabel("Share of images (%)")
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    a.set_title(f"Label is confounded with source (Cramér's V = {V:.2f})", fontweight="bold", pad=24)
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig4_source_label.png"), dpi=160); plt.close()


def fig_properties(fr, out):
    props = [("w", "Width (px)", True), ("kb", "File size (KB)", True),
             ("std", "Intensity SD (contrast)", False), ("entropy", "Histogram entropy (bits)", False)]
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.6))
    for a, (k, t, lg) in zip(ax, props):
        bp = a.boxplot([fr[fr.y == 1][k], fr[fr.y == 0][k]], tick_labels=["COVID-19", "Non-COVID"],
                       patch_artist=True, widths=0.6, showfliers=False)
        for p, cc in zip(bp["boxes"], [C1, C0]):
            p.set_facecolor(cc); p.set_alpha(0.75)
        if lg:
            a.set_yscale("log")
        a.set_title(t, fontsize=10)
    plt.suptitle("Non-diagnostic image properties differ by label", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig5_properties.png"), dpi=160); plt.close()


def fig_shortcut(fr, data_dir, t4, out):
    im = np.asarray(Image.open(os.path.join(data_dir, "images", fr.iloc[5].filename))
                    .convert("L").resize((256, 256)), float) / 255
    b = 40; mb = np.zeros((256, 256), bool); mb[:b] = mb[-b:] = mb[:, :b] = mb[:, -b:] = True
    fig = plt.figure(figsize=(10, 7.2)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.35])
    for i, (t, mk) in enumerate([("(a) Full image", None), ("(b) Centre only (lung field)", mb),
                                 ("(c) Border only (lungs removed)", ~mb)]):
        a = fig.add_subplot(gs[0, i]); v = im.copy()
        if mk is not None:
            v[mk] = 0.5
        a.imshow(v, cmap="gray", vmin=0, vmax=1); a.set_title(t, fontsize=9.5); a.axis("off")
    a = fig.add_subplot(gs[1, :])
    order = ["Permuted labels (chance)", "File properties only (no pixels)", "Centre only (lung field)",
             "4 global intensity statistics", "Border only (lungs removed)", "Full image"]
    r = t4.set_index("condition").loc[order]
    a.barh(r.index, r.auc, xerr=r.auc_sd, capsize=3,
           color=["#AAB7B8", "#8E44AD", "#27AE60", "#8E44AD", "#C0392B", "#2C3E50"])
    for i, v in enumerate(r.auc):
        a.text(v + 0.055, i, f"{v:.3f}", va="center", fontsize=9)
    a.axvline(0.5, c="gray", ls="--", lw=1); a.set_xlim(0.4, 0.9)
    a.set_xlabel("ROC-AUC, COVID-19 vs non-COVID (patient-grouped 5-fold CV x 5 repeats, mean ± SD)")
    a.set_title("(d) The periphery predicts the label better than the lungs do", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig6_shortcut.png"), dpi=160); plt.close()


def fig_duplicates(fr, dups, data_dir, out):
    cross = dups[dups.patient_a != dups.patient_b].copy()
    cross["same_src"] = cross.source_a == cross.source_b
    # one cross-website pair (A) and one same-website pair (B), when available
    cross = pd.concat([cross[~cross.same_src].head(1), cross[cross.same_src].head(1)])
    if len(cross) == 0:
        return
    fig, ax = plt.subplots(1, 2 * len(cross), figsize=(5.5 * len(cross), 3.4)); ax = np.atleast_1d(ax)
    k = 0
    for _, d in cross.iterrows():
        for fn, pid, src in [(d.file_a, d.patient_a, d.source_a), (d.file_b, d.patient_b, d.source_b)]:
            ax[k].imshow(Image.open(os.path.join(data_dir, "images", fn)).convert("L"), cmap="gray")
            ax[k].axis("off"); ax[k].set_title(f"patient ID {pid}\n{src}", fontsize=8.5); k += 1
    plt.suptitle("Provenance defect: the same radiograph registered as different patients", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig7_duplicates.png"), dpi=160); plt.close()


def fig_split(t5, out):
    fig, a = plt.subplots(figsize=(8, 4.4)); yy = np.arange(len(t5))
    for i, (p, q) in enumerate(zip(t5.auc_patient, t5.auc_image)):
        a.plot([p, q], [i, i], c="gray", lw=2, zorder=1)
        a.text(max(p, q) + 0.008, i, f"{q - p:+.3f}", va="center", fontsize=8)
    a.scatter(t5.auc_patient, yy, c="#2C3E50", s=45, label="Patient-grouped split", zorder=2)
    a.scatter(t5.auc_image, yy, c="#E67E22", s=45, label="Image-level split", zorder=2)
    a.set_yticks(yy, t5.model); a.set_xlabel("ROC-AUC"); a.set_xlim(0.5, 0.84)
    a.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    a.set_title("Image-level splitting inflates every model", fontweight="bold", pad=24)
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig8_split.png"), dpi=160); plt.close()


# ------------------------------------------------------------------------ MAIN
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="ds", help="path to the cloned covid-chestxray-dataset repo")
    ap.add_argument("--out", default="results", help="output folder for tables and figures")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    O = lambda f: os.path.join(args.out, f)

    print("[1/5] Ingesting images ...")
    x, P_all = ingest(args.data)

    # Table 1 - dataset profile
    keep = (x.view.isin(FRONTAL) & (x.finding != "todo")).values
    fr = x[keep].reset_index(drop=True); P = P_all[keep]
    fr["y"] = (fr.finding == COVID).astype(int)
    y, groups = fr.y.values, fr.patientid.astype(str).values
    t1 = pd.DataFrame([
        ("X-ray records with an image file (audited set)", len(x)),
        ("Unique patient IDs in audited set", x.patientid.nunique()),
        ("Distinct source websites", x.source.nunique()),
        ("Image formats", ", ".join(f"{v} {k}" for k, v in x.fmt.value_counts().items())),
        ("Colour modes", ", ".join(f"{v} {k}" for k, v in x["mode"].value_counts().items())),
        ("Image width, median (min-max) px", f"{x.w.median():.0f} ({x.w.min()}-{x.w.max()})"),
        ("File size, median (min-max) KB", f"{x.kb.median():.0f} ({x.kb.min():.0f}-{x.kb.max():.0f})"),
        ("Experimental subset images / patients", f"{len(fr)} / {fr.patientid.nunique()}"),
        ("COVID-19 / non-COVID in subset", f"{y.sum()} / {(1 - y).sum()}"),
    ], columns=["item", "value"])
    t1.to_csv(O("table1_dataset_profile.csv"), index=False); print(t1.to_string(index=False))

    print("[2/5] Profiling information quality ...")
    comp = completeness(x); comp.to_csv(O("completeness.csv"))
    rows = []
    for name, col in [("Source website", source_groups(fr)), ("View", fr.view), ("File format", fr.fmt)]:
        V, p = cramers_v(pd.crosstab(col, fr.y)); rows.append((name, p, V))
    t3 = pd.DataFrame(rows, columns=["attribute", "chi2_p", "cramers_v"])
    t3.to_csv(O("table3_label_association.csv"), index=False); print(t3.to_string(index=False))
    props = []
    for c in ["w", "h", "kb", "mean", "std", "entropy", "sat"]:
        props.append((c, fr[fr.y == 1][c].median(), fr[fr.y == 0][c].median(),
                      mannwhitneyu(fr[fr.y == 1][c], fr[fr.y == 0][c]).pvalue))
    pd.DataFrame(props, columns=["property", "median_covid", "median_non", "mannwhitney_p"]) \
        .to_csv(O("image_properties.csv"), index=False)

    print("[3/5] Running shortcut probes (this takes a few minutes) ...")
    X = P.reshape(-1, SIZE, SIZE); mb = border_mask()
    file_feats = fr[["w", "h", "kb"]].assign(ar=fr.w / fr.h, png=(fr.fmt == "PNG").astype(int),
                                               rgb=(fr["mode"] != "L").astype(int)).values
    conditions = {
        "Full image": (X.reshape(len(X), -1), pixel_model),
        "Border only (lungs removed)": (X[:, mb], pixel_model),
        "Centre only (lung field)": (X[:, ~mb], pixel_model),
        "4 global intensity statistics": (fr[["mean", "std", "entropy", "sat"]].values, tabular_model),
        "File properties only (no pixels)": (file_feats, tabular_model),
    }
    t4_rows, t5_rows = [], []
    for name, (F, mk) in conditions.items():
        pa = cross_validate(F, y, groups, mk, "patient")
        im = cross_validate(F, y, groups, mk, "image")
        t4_rows.append((name, *pa)); t5_rows.append((name, pa[0], im[0]))
        print(f"  {name:34s} AUC {pa[0]:.3f} ± {pa[1]:.3f}")
    rng = np.random.default_rng(0); aucs = []
    for seed in range(N_REPEATS):
        yp = rng.permutation(y)
        for tr, te in splitter("patient", seed).split(X, yp, groups):
            F = X.reshape(len(X), -1); clf = pixel_model().fit(F[tr], yp[tr])
            aucs.append(roc_auc_score(yp[te], clf.predict_proba(F[te])[:, 1]))
    t4_rows.append(("Permuted labels (chance)", np.mean(aucs), np.std(aucs), np.nan, np.nan))
    t4 = pd.DataFrame(t4_rows, columns=["condition", "auc", "auc_sd", "bacc", "bacc_sd"])
    t4.to_csv(O("table4_shortcut_probes.csv"), index=False)

    # Source identification from the border frame
    src = source_groups(fr).values; accs = []
    for tr, te in StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=0).split(X, src, groups):
        F = X[:, mb]; clf = pixel_model().fit(F[tr], src[tr])
        accs.append(balanced_accuracy_score(src[te], clf.predict(F[te])))
    print(f"  Source ID from border: balanced acc {np.mean(accs):.3f} (chance {1 / len(set(src)):.3f})")

    print("[4/5] Evaluation leakage + duplicates ...")
    Z = (P - P.mean(1, keepdims=True)) / P.std(1, keepdims=True)
    knn = lambda: KNeighborsClassifier(1, metric="cosine")
    t5_rows.append(("Full image (1-NN, memorising)",
                    cross_validate(Z, y, groups, knn, "patient")[0],
                    cross_validate(Z, y, groups, knn, "image")[0]))
    t5 = pd.DataFrame(t5_rows, columns=["model", "auc_patient", "auc_image"])
    t5["delta_leak"] = t5.auc_image - t5.auc_patient
    t5.to_csv(O("table5_split_inflation.csv"), index=False); print(t5.round(3).to_string(index=False))

    C = Z @ Z.T / Z.shape[1]; np.fill_diagonal(C, 0)
    i, j = np.where(np.triu(C > 0.99))
    dups = pd.DataFrame(dict(r=C[i, j], file_a=fr.filename.values[i], file_b=fr.filename.values[j],
                             patient_a=groups[i], patient_b=groups[j],
                             source_a=fr.source.values[i], source_b=fr.source.values[j],
                             label_conflict=y[i] != y[j]))
    dups.to_csv(O("duplicates.csv"), index=False)
    print(f"  near-duplicate pairs: {len(dups)}, cross-patient: {(dups.patient_a != dups.patient_b).sum()}")

    print("[5/5] Drawing figures ...")
    fig_samples(fr, args.data, args.out)
    fig_completeness(comp, args.out)
    fig_source_label(fr, t3.cramers_v[0], args.out)
    fig_properties(fr, args.out)
    fig_shortcut(fr, args.data, t4, args.out)
    fig_duplicates(fr, dups, args.data, args.out)
    fig_split(t5, args.out)
    json.dump(dict(source_id_bacc=float(np.mean(accs)), source_id_sd=float(np.std(accs))),
              open(O("source_identification.json"), "w"), indent=2)
    print(f"Done. Results in ./{args.out}/")


if __name__ == "__main__":
    main()
