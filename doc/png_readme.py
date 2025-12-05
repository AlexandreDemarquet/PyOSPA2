import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ospa2 import OSPA2

# --- Simple synthetic example ---


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
# --- Compute OSPA² over time ---

o = OSPA2(c=10, p=1, q=1, window_length=3, cols=["x","y"], id_col="track_id")
times, o2_vals, loc_vals, card_vals = o.ospa2_over_time(gt, trk)

# --- Plotly visualization ---

fig = go.Figure()
fig.add_trace(go.Scatter(x=times, y=o2_vals, mode="lines+markers", name="OSPA2"))
fig.add_trace(go.Scatter(x=times, y=loc_vals, mode="lines+markers", name="Localization"))
fig.add_trace(go.Scatter(x=times, y=card_vals, mode="lines+markers", name="Cardinality"))

fig.update_layout(
    title="OSPA² Score Over Time (Simple Example)",
    xaxis_title="Time",
    yaxis_title="Score",
    template="plotly_white"
)

fig.write_image("ospa2_example.png")
# fig.show()
