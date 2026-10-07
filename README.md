# IQ-Audit — code for "Seeing Is Not Believing"
<img width="100%" height="199" alt="Screenshot 2026-10-06 at 20 37 27" src="https://github.com/user-attachments/assets/a070ac29-c358-4c0c-ae10-27f70beb2ecc" />


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
| `iq_audit_minimal.py` | Short version printed in Appendix A of the paper (Tables 3–4, splits, duplicates) |
| `requirements.txt` | Python dependencies |
| `results/` | Outputs from our run on 30 Sep 2026 (tables as CSV, figures as PNG) |
