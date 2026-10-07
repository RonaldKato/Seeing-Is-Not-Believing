# IQ-Audit — code for "Seeing Is Not Believing"

Reproduces every table and figure in the paper *Seeing Is Not Believing: An Information
Quality Audit of Open Medical Imaging Repositories*.

## Quick start

```bash
git clone https://github.com/ieee8023/covid-chestxray-dataset.git ds   # ~530 MB open dataset
pip install -r requirements.txt
python iq_audit.py --data ds --out results                           # ~5-10 min on a laptop
```

## Files

| File | What it does |
|---|---|
| `iq_audit.py` | Full pipeline: ingest → IQ profile → shortcut probes → leakage test → Figures 2–8 + CSV tables |
| `architecture_figure.py` | Draws Figure 1, the research architecture |
| `iq_audit_minimal.py` | Short version printed in Appendix A of the paper (Tables 3–4, splits, duplicates) |
| `requirements.txt` | Python dependencies |
| `results/` | Outputs from our run on 30 Sep 2026 (tables as CSV, figures as PNG) |

## Output → paper mapping

| Output | Paper |
|---|---|
| `table1_dataset_profile.csv` | Table 1 |
| `completeness.csv`, `fig3_completeness.png` | Section 5.1, Figure 3 |
| `table3_label_association.csv`, `fig4_source_label.png` | Table 3, Figure 4 |
| `image_properties.csv`, `fig5_properties.png` | Section 5.2, Figure 5 |
| `table4_shortcut_probes.csv`, `fig6_shortcut.png` | Table 4, Figure 6 |
| `source_identification.json` | Section 5.3 (source ID from border) |
| `duplicates.csv`, `fig7_duplicates.png` | Section 5.4, Figure 7 |
| `table5_split_inflation.csv`, `fig8_split.png` | Section 5.5, Figure 8 |
| `fig1_architecture.png` | Figure 1 (research architecture) |
| `fig2_samples.png` | Figure 2 |

Results use fixed random seeds, so reruns match the paper. The dataset is updated occasionally
upstream; a newer snapshot may shift numbers slightly.

## Extending it

- Change `BORDER` (default 10 px of 64) to test other frame widths.
- Swap `pixel_model()` for a pretrained CNN to run deep-model probes (paper's future work).
- Point `--data` at any repository with a `metadata.csv` + `images/` layout after adapting
  the column names in `ingest()`.
