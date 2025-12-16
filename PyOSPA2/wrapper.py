import numpy as np
import pandas as pd
try:
    from ._ospa2 import compute_distance_matrix, ospa2_from_matrix, average_ospa2_at_time
except ImportError as e:
    raise ImportError("The _ospa2 extension module is missing. Build the package first.") from e

class OSPA2:
    """High-level OSPA2 calculator (Python wrapper around C++ backend).

    This class prepares trajectories (DataFrame -> lists of contiguous NumPy arrays),
    calls the C++ backend to compute the pairwise distance matrix, and computes the
    aggregated OSPA(2) score.

    Constructor parameters:
    - c (float): distance cap/threshold (default 100.0)
    - p (int): order (default 1)
    - q (int): order of the base distance (default 1)
    - window_length (int): sliding window length in frames (>=1)
    - cols (List[str] or None): DataFrame columns to use as coordinates (default ['x','y'])
    - id_col (str): column name for track identifier (default 'track_id')
    - time_col (str): column name for time index (default 'ts')
    - weight_col (str|None): optional weight column (not used by current implementation)
    """

    def __init__(self, c=100.0, p=1, q=1, window_length=10, cols=None, id_col='track_id', time_col='ts', weight_col=None):
        """Initialize an OSPA2 calculator.

        Args:
            c (float): distance cap/threshold (must be > 0).
            p (int): order p (must be > 0).
            q (int): order of the base distance q (must be > 0).
            window_length (int): sliding window size (>= 1).
            cols (list[str] | None): columns containing coordinates (default ['x','y']).
            id_col (str): track identifier column name.
            time_col (str): time column name.
            weight_col (str | None): optional weight column (not currently used).

        Raises:
            ValueError: if `window_length` < 1 or if `c`, `p`, `q` are not strictly positive.
        """
        if window_length < 1:
            raise ValueError("window_length must be >= 1")
        if c <= 0 or p <= 0 or q <= 0:
            raise ValueError("c, p and q must be > 0")
        self.c = c
        self.p = p
        self.q = q
        self.window_length = window_length
        self.cols = cols if cols is not None else ['x','y']
        self.id_col = id_col
        self.time_col = time_col
        self.weight_col = weight_col

    def _ensure_contiguous(self, arr: np.ndarray) -> np.ndarray:
        """Ensure `arr` is a C-contiguous `np.ndarray` of dtype `float64`.

        Converts to `float64` if needed (using `astype(..., copy=False)`). Returns the same
        object if already compliant, or a contiguous copy/view otherwise.
        """
        if arr.dtype != np.float64:
            arr = arr.astype(np.float64, copy=False)
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        return arr

    def df_to_trajs(self, df: pd.DataFrame, t: int):
        """Extract trajectories from a DataFrame for the window ending at time `t`.

        Selects rows where `time_col` is in [t-window_length+1, t], groups by `id_col`, sorts by
        `time_col` and converts the columns in `cols` into contiguous `np.ndarray` of dtype `float64`.

        Args:
            df (pd.DataFrame): DataFrame containing at least `time_col`, `id_col`, and the columns in `cols`.
            t (int): final time of the window.

        Returns:
            list[np.ndarray]: list of arrays shaped (L_i, D) for each track id. Returns an empty list if no data.
        """
        t0 = t - self.window_length + 1
        w = df[(df[self.time_col] >= t0) & (df[self.time_col] <= t)]
        if w.empty:
            return []
        trajs = []
        w = w.drop_duplicates()
        for tid, g in w.groupby(self.id_col):
            g = g.sort_values(self.time_col)
            # arr = g[self.cols].to_numpy(dtype=np.float64, copy=False)
            # arr = self._ensure_contiguous(arr)  
            # trajs.append(arr)
            trajs.append(g[['x', 'y']].values)

        return trajs

    def ospa2_at_time(self, gt_df, trk_df, t):
        """Compute OSPA2 (global), localization and cardinality for the window ending at `t`.

        Behavior:
        - Converts GT and TRK into lists of trajectories via `df_to_trajs`.
        - If both lists are empty -> (0.0, 0.0, 0.0).
        - If one side is empty -> compute cardinality-only contribution.
        - Otherwise: call `compute_distance_matrix(gt_trajs, trk_trajs, c, q)` then
          `ospa2_from_matrix(D, c, p)`.

        Args:
            gt_df (pd.DataFrame): ground-truth DataFrame.
            trk_df (pd.DataFrame): tracks DataFrame.
            t (int): final time of the window.

        Returns:
            tuple(float, float, float): (ospa2, loc, card)
        """
        gt_trajs = self.df_to_trajs(gt_df, t)
        trk_trajs = self.df_to_trajs(trk_df, t)

        c, p = self.c, self.p        
        m, n = len(gt_trajs), len(trk_trajs)

        if m == 0 and n == 0:
            return 0.0, 0.0, 0.0
        if m == 0 or n == 0:
            card = (c ** p * abs(m - n) / max(m, n)) ** (1/p)
            return float(card), 0.0, float(card)

        D = compute_distance_matrix(gt_trajs, trk_trajs, self.c, self.q)
        ospa2, loc, card = ospa2_from_matrix(D, self.c, self.p)
        return float(ospa2), float(loc), float(card)

    def ospa2_over_time(self, gt_df, trk_df):
        """Compute OSPA2 over time for each unique timestamp in `gt_df`.

        By default, evaluated timestamps are `sorted(gt_df[self.time_col].unique())`.
        To evaluate the union of timestamps from GT and TRK, change the caller.

        Args:
            gt_df (pd.DataFrame): ground-truth DataFrame.
            trk_df (pd.DataFrame): tracks DataFrame.

        Returns:
            tuple(list[int], list[float], list[float], list[float]):
                (ts_list, o2, o2_loc, o2_card)
        """
        ts_list = sorted(gt_df[self.time_col].unique())
        o2, o2_loc, o2_card = [], [], []
        for t in ts_list:
            ospa2, loc, card = self.ospa2_at_time(gt_df, trk_df, int(t))
            o2.append(ospa2)
            o2_loc.append(loc)
            o2_card.append(card)
        return ts_list, o2, o2_loc, o2_card

    def average_ospa2_over_time(self, l_gt_df: list[pd.DataFrame], l_trk_df: list[pd.DataFrame]) -> tuple[list[int], list[float], list[float], list[float]]:
        """Compute average OSPA2 over time for a list of ground-truth and track DataFrames.
        
        Uses parallel C++ computation for averaging over pairs at each timestamp.
        
        Args:
            l_gt_df: List of ground-truth DataFrames.
            l_trk_df: List of track DataFrames.
        
        Returns:
            tuple(list[int], list[float], list[float], list[float]):
                (ts_list, avg_o2, avg_o2_loc, avg_o2_card)
        """
        assert len(l_gt_df) == len(l_trk_df), "Lists of DataFrames must have the same length."
        assert len(l_gt_df) > 0, "Input lists must not be empty."
        # Get timestamps from first GT DataFrame
        ts_list = sorted(l_gt_df[0][self.time_col].unique())
        
        avg_o2 = []
        avg_o2_loc = []
        avg_o2_card = []
        
        for t in ts_list:
            # Prepare pairs of trajectories for this timestamp
            pairs = []
            for gt_df, trk_df in zip(l_gt_df, l_trk_df):
                gt_trajs = self.df_to_trajs(gt_df, t)
                trk_trajs = self.df_to_trajs(trk_df, t)
                pairs.append((gt_trajs, trk_trajs))
            
            # Call C++ function for parallel averaging
            ospa2, loc, card = average_ospa2_at_time(pairs, self.c, self.p, self.q)
            avg_o2.append(ospa2)
            avg_o2_loc.append(loc)
            avg_o2_card.append(card)
        
        return ts_list, avg_o2, avg_o2_loc, avg_o2_card