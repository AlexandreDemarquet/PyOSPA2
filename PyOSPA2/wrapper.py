import numpy as np
import pandas as pd
try:
    from ._ospa2 import compute_distance_matrix, ospa2_from_matrix
except ImportError as e:
    raise ImportError("The _ospa2 extension module is missing. Build the package first.") from e

class OSPA2:
    """High-level OSPA2 calculator using C++ backend."""

    def __init__(self, c=100.0, p=1.0, q=1.0, window_length=10, cols=None, id_col='track_id', time_col='ts', weight_col=None):
        self.c = c
        self.p = p
        self.q = q
        self.window_length = window_length
        self.cols = cols if cols is not None else ['x','y']
        self.id_col = id_col
        self.time_col = time_col
        self.weight_col = weight_col

    def _ensure_contiguous(self, arr: np.ndarray) -> np.ndarray:
        if arr.dtype != np.float64:
            arr = arr.astype(np.float64, copy=False)
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        return arr

    def df_to_trajs(self, df: pd.DataFrame, t: int):
        t0 = t - self.window_length + 1
        w = df[(df[self.time_col] >= t0) & (df[self.time_col] <= t)]
        if w.empty:
            return []
        trajs = []
        w = w.drop_duplicates()
        groups = w.groupby(self.id_col)
        for tid, g in groups:
            g = g.sort_values(self.time_col)
            # arr = g[self.cols].to_numpy(dtype=np.float64, copy=False)
            # arr = self._ensure_contiguous(arr)  
            # trajs.append(arr)
            trajs.append(g[['x', 'y']].values)

        return trajs

    def ospa2_at_time(self, gt_df, trk_df, t):
        gt_trajs = self.df_to_trajs(gt_df, t)
        trk_trajs = self.df_to_trajs(trk_df, t)

        c, p = self.c, self.p        
        m, n = len(gt_trajs), len(trk_trajs)

        if m == 0 and n == 0:
            return 0, 0, 0
        if m == 0 or n == 0:
            card = (c ** p * abs(m - n) / max(m, n)) ** (1/p)
            return card, 0, card

        D = compute_distance_matrix(gt_trajs, trk_trajs, self.c, self.q)
        ospa2, loc, card = ospa2_from_matrix(D, self.c, self.p)
        return ospa2, loc, card

    def ospa2_over_time(self, gt_df, trk_df):
        ts_list = sorted(gt_df[self.time_col].unique())
        o2, o2_loc, o2_card = [], [], []
        for t in ts_list:
            ospa2, loc, card = self.ospa2_at_time(gt_df, trk_df, int(t))
            o2.append(ospa2)
            o2_loc.append(loc)
            o2_card.append(card)
        return ts_list, o2, o2_loc, o2_card
