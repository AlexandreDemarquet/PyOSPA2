Voici le **README complet en Markdown**, prêt à copier-coller dans GitHub, incluant la **citation officielle OSPA(2)**.

---

# 📘 OSPA2 — Fast OSPA(2) Metric with AVX2-Optimized LAPJV Solver

**OSPA2** is a high-performance C++/Python library implementing the
**OSPA(2) metric for multi-target tracking**, based on:

* **LAPJV (Jonker–Volgenant Hungarian Algorithm)**
  from: [https://github.com/src-d/lapjv/tree/master](https://github.com/src-d/lapjv/tree/master)
  (adapted and compiled with **AVX2** support)
* **Eigen 3.4.0** for vectorized linear algebra
* **Pybind11** for seamless Python bindings

It computes the OSPA(2) metric over time using sliding windows, directly from Pandas DataFrames.

---

## ✨ Features

* ⚡ **Very fast** assignment via AVX2-optimized LAPJV
* 🧮 Full OSPA(2) support: parameters `c`, `p`, `q`, time window length
* 🧩 Accepts Pandas DataFrames (with id + feature columns)
* 🔗 Easy Python API
* 🧪 Tested with `pytest`
* 📦 Simple installation: `pip install .`

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourname/ospa2.git
cd ospa2
```

### 2. Install the package

```bash
pip install .
```

This builds the C++ extension including LAPJV and Eigen.

---

## 📁 Project Structure

```
ospa2/
│
├── cpp/
│   ├── lapjv/               # AVX2 LAPJV implementation (included)
│   ├── ospa2.cpp            # OSPA2 core: distance matrix + OSPA2 metric
│   ├── bindings.cpp         # pybind11 module
│   └── CMakeLists.txt       # build rules
│
├── python/
│   └── ospa2.py             # high-level Python API
│
├── tests/
│   └── test_cpp_core.py     # unit tests (pytest)
│
└── setup.py                 # setuptools build config
```

---




## 📊 Usage OSPA² Example (Deterministic Values)

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ospa2 import OSPA2

# Time steps
ts = np.array([0, 1, 2, 3, 4])

# ============================================================
# 1) Ground Truth (2 objects)
# ============================================================

gt = pd.DataFrame({
    "ts": np.repeat(ts, 2),
    "track_id": np.tile([1, 2], len(ts)),
    "x": [
        # Object 1
        0.0,  1.0,  2.0,  3.0,  4.0,
        # Object 2
        10.0, 11.0, 12.0, 13.0, 14.0
    ],
    "y": [
        # Object 1
        0.0,  0.0,  0.0,  0.0,  0.0,
        # Object 2
        5.0,  5.0,  5.0,  5.0,  5.0
    ],
})

# ============================================================
# 2) Tracker Output (3 objects) — NO RANDOM
# ============================================================

trk = pd.DataFrame({
    "ts": np.repeat(ts, 3),
    "track_id": np.tile([1, 2, 3], len(ts)),

    "x": [
        # Track 1 (correct but slightly offset)
        0.2, 1.1, 2.2, 3.3, 4.3,
        # Track 2 (correct but slightly offset)
        10.1, 11.2, 12.2, 13.4, 14.5,
        # Track 3 (false positive near object 2)
        9.0,  9.5, 10.0, 10.5, 11.0
    ],

    "y": [
        # Track 1
        0.1, 0.1, 0.2, 0.2, 0.3,
        # Track 2
        5.1, 5.2, 5.1, 5.2, 5.3,
        # Track 3 (false positive)
        4.8, 4.9, 5.0, 5.1, 5.2
    ],
})

# ============================================================
# 3) Compute OSPA²
# ============================================================

o = OSPA2(c=10, p=2, q=1, window_length=3, cols=["x", "y"], id_col="track_id")
times, ospa2_vals, loc_vals, card_vals = o.ospa2_over_time(gt, trk)

# ============================================================
# 4) Plot the OSPA² components
# ============================================================

fig = go.Figure()
fig.add_trace(go.Scatter(x=times, y=ospa2_vals, mode="lines+markers", name="OSPA²"))
fig.add_trace(go.Scatter(x=times, y=loc_vals, mode="lines+markers", name="Localization Error"))
fig.add_trace(go.Scatter(x=times, y=card_vals, mode="lines+markers", name="Cardinality Error"))

fig.update_layout(
    title="OSPA² Over Time (Manual Deterministic Example)",
    xaxis_title="Time",
    yaxis_title="Score",
    template="plotly_white"
)

fig.write_image("ospa2_example.png")
```


This example shows:

- **2 ground truth objects**
- **3 tracker objects**, including **one persistent false positive**
- Slight offsets on the correct tracks → **localization error**
- One extra track → **cardinality error**

The resulting OSPA² components are plotted below:

![OSPA2 Example](ospa2_example.png)

---

## 🧪 Testing

Simply run:

```bash
pytest -q
```

---

## 📚 References & Citation

### **Original OSPA(2) Metric**

> Schuhmacher, D., Vo, B.-T., & Vo, B.-N. (2008).
> **A Statistic for Performance Evaluation of Multi-Object Filters**.
> *IEEE Transactions on Signal Processing*, 56(8), 3447–3457.
> [https://doi.org/10.1109/TSP.2008.924688](https://doi.org/10.1109/TSP.2008.924688)

### **Original OSPA (base metric)**

> R. Mahler, “Multitarget Bayes filtering and the PHD filter,”
> in *AeroSense'03*, SPIE, 2003.

### **LAPJV Solver**

> R. Jonker & A. Volgenant,
> “A shortest augmenting path algorithm for dense and sparse linear assignment problems,”
> *Computing*, 38, 325–340, 1987.

---

## 📝 License

MIT License (or your choice).

---

## 🙌 Contributions

Open to issues, pull requests, and performance improvements.


