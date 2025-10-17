# MXene_seg   

<div align="center">

| Notebooks | Python | License |
|-----------|--------|---------|
| 0–3 (training → theory) | 3.9 | MIT |

</div>

---

## 1  Project Overview
`MXene_seg` is a self-contained repository for
* finetuning deep-learning models on STEM / HAADF images (`functions/finetuning_training.py`)
* detecting lattice defects & computing vacancy statistics (`functions/finding_defects.py`)
* interactive 3-D visualisation of layer-projected atoms (`functions/layers.py`, `functions/three_d.py`)

All heavy-lifting code lives in **`functions/`**; the numbered folders hold Jupyter notebooks that document the full workflow.

---

## 2  Quick Start

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


