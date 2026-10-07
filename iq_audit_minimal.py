import os, numpy as np, pandas as pd
from PIL import Image
from urllib.parse import urlparse
from scipy.stats import chi2_contingency
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
D = "ds"
m = pd.read_csv(f"{D}/metadata.csv")
x = m[(m.modality == "X-ray") & (m.folder == "images")].copy()
x["source"] = x.url.fillna("").map(lambda u: urlparse(u).netloc.replace("www.", ""))
feats, px = [], []
for fn in x.filename:
    im = Image.open(f"{D}/images/{fn}")
    g = np.asarray(im.convert("L"), np.float32) / 255
    feats.append(dict(w=im.width, h=im.height, png=int(im.format == "PNG"),
                      rgb=int(im.mode != "L"), std=g.std(), kb=os.path.getsize(f"{D}/images/{fn}") / 1024))
    px.append(np.asarray(im.convert("L").resize((64, 64), Image.BILINEAR), np.float32).ravel() / 255)
x = pd.concat([x.reset_index(drop=True), pd.DataFrame(feats)], axis=1)
P = np.stack(px)
print((x.notna().mean() * 100).round(1).sort_values())
keep = x.view.isin(["PA", "AP", "AP Supine", "AP Erect"]) & (x.finding != "todo")
fr, P = x[keep].reset_index(drop=True), P[keep.values]
y = (fr.finding == "Pneumonia/Viral/COVID-19").astype(int).values
groups = fr.patientid.astype(str).values
top = fr.source.value_counts().index[:6]
ct = pd.crosstab(fr.source.where(fr.source.isin(top), "other"), y)
chi2 = chi2_contingency(ct)[0]
print("Cramer's V (source):", np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))
X = P.reshape(-1, 64, 64); b = 10
border = np.zeros((64, 64), bool); border[:b] = border[-b:] = border[:, :b] = border[:, -b:] = True
inputs = {"full": X.reshape(len(X), -1), "centre": X[:, ~border], "border": X[:, border],
          "file": fr[["w", "h", "png", "rgb", "kb"]].assign(ar=fr.w / fr.h).values}
def model(name):
    if name == "file":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    return make_pipeline(StandardScaler(), PCA(50, random_state=0),
                         LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
for name, F in inputs.items():
    for split in ["patient", "image"]:
        aucs = []
        for seed in range(5):
            cv = (StratifiedGroupKFold(5, shuffle=True, random_state=seed) if split == "patient"
                  else StratifiedKFold(5, shuffle=True, random_state=seed))
            for tr, te in cv.split(F, y, groups):
                clf = model(name).fit(F[tr], y[tr])
                aucs.append(roc_auc_score(y[te], clf.predict_proba(F[te])[:, 1]))
        print(f"{name:7s} {split:8s} AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
Z = (P - P.mean(1, keepdims=True)) / P.std(1, keepdims=True)
C = Z @ Z.T / Z.shape[1]; np.fill_diagonal(C, 0)
i, j = np.where(np.triu(C > 0.99))
print("duplicate pairs:", len(i), "cross-patient:", (groups[i] != groups[j]).sum())
