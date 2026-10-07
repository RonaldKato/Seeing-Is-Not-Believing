#IQ-Audit — Seeing Is Not Believing
<img width="600" height="237" alt="IQ-Audit Architecture" src="https://github.com/user-attachments/assets/6cb8140a-378e-410e-856a-172d773f566a" />

An Information Quality Audit of Open Medical Imaging Repositories
IQ-Audit is an open, reproducible Python framework for auditing the information quality of medical imaging repositories and testing whether seemingly harmless data deficiencies become learnable shortcuts for AI models.
The project connects Information Systems (IS) information-quality theory, medical imaging, machine learning, and AI governance through a practical audit pipeline:
Repository → Quality → Shortcuts → Leakage → Evidence

#Why IQ-Audit?
Medical images may appear clinically meaningful while carrying hidden signals from source websites, image borders, file properties, duplication, metadata gaps, or acquisition practices. Machine-learning models can exploit these signals and achieve impressive performance without learning the intended clinical concept.
IQ-Audit makes these risks measurable.
Key Findings
Applied to the COVID-19 Image Data Collection:
Audit finding	Result
Chest X-rays	866
Source websites	38
Metadata fields ≥90% complete	5 / 19
Label–source confounding	Cramér’s V = 0.54
Duplicate radiographs	14
Border-only ROC-AUC	0.729
Full-image ROC-AUC	0.752
Lung-only ROC-AUC	0.661
File-properties-only ROC-AUC	0.647
Leakage-related AUC increase	Up to 0.045


#The central insight
Poor information quality can make AI performance look better.

A classifier using only image boundaries—while excluding the lungs—achieved an AUC of 0.729, approaching the full-image performance of 0.752. This demonstrates how repository-level deficiencies can become predictive shortcuts.
We describe this as the IQ Paradox of Learning Systems: information that is poor from a clinical or informational perspective may still be highly valuable to a machine-learning model.
What IQ-Audit Does
1. Ingest
Loads images and associated repository metadata.
2. Profile Information Quality
Measures completeness, consistency, objectivity, provenance, and accessibility.
3. Detect Shortcuts
Tests whether non-diagnostic image regions and metadata can predict diagnostic labels.
4. Test Leakage
Compares patient-grouped evaluation with image-level splitting.
5. Generate Evidence
Produces reproducible figures, tables, metrics, and audit outputs for researchers and dataset curators.
Quick Start
git clone https://github.com/ieee8023/covid-chestxray-dataset.git ds
pip install -r requirements.txt
python iq_audit.py --data ds --out results

The pipeline typically runs in 5–10 minutes on a laptop after the dataset is available.
Repository Structure
File	Purpose
iq_audit.py	Complete IQ-Audit pipeline: profiling, shortcut probes, leakage testing, figures, and CSV outputs
iq_audit_minimal.py	Compact implementation corresponding to the paper's Appendix A
requirements.txt	Python dependencies
results/	Reproduced audit tables and figures


Research Contribution
IQ-Audit moves information-quality assessment beyond “Is this dataset documented?” toward a more consequential question:
“Can the information we consider non-diagnostic actually be learned by an AI model?”

The framework provides an empirical bridge between information quality, data provenance, machine-learning behaviour, and AI governance.
Intended Users
- Medical AI researchers — identify hidden shortcuts before model development.
- Dataset curators — quantify repository-level quality problems.
- Reviewers & editors — assess whether reported performance may reflect leakage or confounding.
- Healthcare organisations — strengthen AI and dataset due diligence.
- IS researchers — study information quality in machine-learning information environments.
Reproducibility
All analysis scripts, dependencies, generated tables, and figures are provided to support transparent and reproducible auditing.
Paper: Seeing Is Not Believing: An Information Quality Audit of Open Medical Imaging Repositories
Framework: IQ-Audit
Language: Python
Focus: Information Quality · Medical Imaging · Shortcut Learning · Data Provenance · AI Governance · Design Science
