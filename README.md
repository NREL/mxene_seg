# MXene Segmentation and 3D Clustering    

<div align="center">

| Notebooks | Python | License |
|-----------|--------|---------|
| 0–3 (training → theory) | 3.9 | BSD-3 |

</div>

---

## 1  Project Overview
`MXene_seg` is a self-contained repository for
* finetuning deep-learning models on STEM / HAADF images (`functions/finetuning_training.py`)
* detecting lattice defects & computing vacancy statistics (`functions/finding_defects.py`)
* interactive 3-D visualisation of layer-projected atoms (`functions/layers.py`, `functions/three_d.py`)

All heavy-lifting code lives in **`functions/`**; the numbered folders hold Jupyter notebooks that document the full workflow.

## 2 Contact Information
This package was developed by Grace Guinan, Michelle A. Smeaton, Brian C. Wyatt, Steven Goldy, Hilary Egan, Andrew
Glaws, Garritt J. Tucker, Babak Anasori and Steven R. Spurgeon. Address all questions to: steven.spurgeon@nrel.gov

Copyright (c) 2025 National Laboratory of the Rockies (NLR)

NLR Software Record SWR-25-67

## 3 How to Cite
Please cite our Arxiv preprint: Guinan, G., Smeaton, M. A., Wyatt, B. C., Goldy, S., Egan, H., Glaws, A., Tucker, G. J., Anasori, B., & Spurgeon, S. R. (2025). Revealing the hidden third dimension of point defects in two-dimensional MXenes. arXiv. https://arxiv.org/abs/2511.08350

---

## 3  Quick Start

```bash
# ❶ Clone the repo
git clone https://github.com/<your-name>/MXene_seg.git
cd MXene_seg

# ❷ Create & activate a local virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\activate

# ❸ Install runtime dependencies (+ your helper package)
pip install -r requirements.txt
pip install -e .



MXene_seg/
├── functions/            # reusable Python modules
│   ├── __init__.py
│   ├── finetuning_training.py
│   ├── finding_defects.py
│   ├── layers.py
│   └── three_d.py  
├── 0_training/           # notebooks step 0
├── 1_defect_detecting/
├── 2_three_dimensional/
├── 3_theory/
├── data
├── requirements.txt
├── LICENSE.md
└── README.md             # ← you are here


